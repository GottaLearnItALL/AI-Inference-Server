## Concurrency

* Running Curl A = T time at 52 tok/s
* RUnning Curl B = 2T time at 52 tok/s
* T_1 = Elapsed wall-clock time: 6.732448 seconds, for one curl


Model = Qwnen3.5
Concurrency 1: {
    'prompt': "Explain what a quantum computer.", "max_tokens": 200",
    'generated_time': 5.15s,
    'tok/s requested': 38.8,
    'aggregate_token': 38.8,

}


Concurrency 2: {
    'prompt': ""Explain what a deadlock is."", "max_tokens": 200",
    'generated_time': 9.66/9.43,
    'tok/s requested': 20.7,
    'aggregate_token': 41.4,

}


TimeToFirstToken = 0.6585s 7.8s better latency than total work

# Async Client Request:
'ttft': 0.5331447500502691
total': 5.774575000046752
'tokens': 200


# MBU = (2GB model sixe * 42 tok/s throughput) / 120GB bandwidth = 70% MBU
# during decode

