import numpy as np
from random import Random
from constants import *


DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
NOISE_LOW = 0.9
NOISE_HIGH = 1.1


# class that simulates video chunk download
class Environment:
    def __init__(self, traces, type, video, random_seed=RANDOM_SEED):
        self.myRandom = Random(random_seed)

        self.all_cooked_time = traces[TIME]
        self.all_cooked_bw = traces[BANDWIDTH]

        self.video_chunk_counter = 0
        self.buffer_size = video[VIDEO_CHUNK_LENGTH] * 1000.0

        # pick a random trace file
        self.type = type

        if self.type == TEST:
            self.trace_idx = 0
        if self.type == TRAIN:
            self.trace_idx = self.myRandom.randint(0, len(self.all_cooked_time) - 1)
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        if self.type == TEST:
            self.mahimahi_ptr = 1
        if self.type == TRAIN:
            self.mahimahi_ptr = self.myRandom.randint(1, len(self.cooked_bw) - 1)
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video = video


    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < len(self.video[BITRATE_LEVELS])

        video_chunk_size = self.video[VIDEO_SIZES][self.video_chunk_counter][quality] * B_IN_MB

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= self.myRandom.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += self.video[VIDEO_CHUNK_LENGTH] * M_IN_K

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_CAPACITY * M_IN_K:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_CAPACITY * M_IN_K
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.video[NUM_OF_CHUNKS] - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.video[NUM_OF_CHUNKS]:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            if self.type == TEST:
                self.trace_idx = 0
            if self.type == TRAIN:
                self.trace_idx = self.myRandom.randint(0, len(self.all_cooked_time) - 1)
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # note: trace file starts with time 0
            if self.type == TEST:
                self.mahimahi_ptr = 1
            if self.type == TRAIN:
                self.mahimahi_ptr = self.myRandom.randint(1, len(self.cooked_bw) - 1)
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]


        state = np.zeros(FLAT_INPUT_LEN)
        state[LAST_BR_IDX] = quality
        state[BUFFER_IDX] = return_buffer_size / MILLISECONDS_IN_SECOND
        state[THROUGHPUT_IDX] = video_chunk_size / delay    # in kbyte ps
        state[DELAY_IDX] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[NEXT_CHUNKS_START_IDX: NEXT_CHUNKS_START_IDX + self.video[BR_DIM]] = \
              np.array(self.video[VIDEO_SIZES][self.video_chunk_counter])    # in mbyte
        state[NEXT_CHUNKS_START_IDX + self.video[BR_DIM]] = np.minimum(video_chunk_remain, self.video[NUM_OF_CHUNKS])

        return state, sleep_time, rebuf / M_IN_K, end_of_video
