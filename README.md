
```text
mmlu-limit10:
- smollm2-135M  0.26
- smollm2-360M  0.28
- Qwen3-0.6B    0.49
- smollm2-1.7B  0.52
- Qwen3-1.7B    0.64
- smollm3-3B    0.64

hellaswag-limit1000(acc & acc norm):
- smollm2-135M  0.376   0.453
- Qwen3-0.6B    0.406   0.482
- smollm2-360M  0.414   0.535
- Qwen3-1.7B    0.447   0.546
- smollm2-1.7B  0.484   0.636
- smollm3-3B    0.521   0.671

=== PLL Summary (over cc_news) ===
Model                                         | mean ± std
-------------------------------------------------------------
HuggingFaceTB/SmolLM2-135M                    | 2.5990 ± 1.8230
HuggingFaceTB/SmolLM2-360M                    | 2.6107 ± 1.6664
nickypro/tinyllama-42M                        | 3.0176 ± 0.8496
Mostafa8Mehrabi/qwen3-30m-tinystories-final   | 3.6821 ± 1.1920
-------------------------------------------------------------
HUMAN                                         | 1.5283 ± 0.5924


=== BoolQ Accuracy Summary ===
Model                                         | Acc% | Match% | Acc@Match% | Correct/Total | Matched | Unmatched
---------------------------------------------------------------------------------------------------
Qwen/Qwen3-0.6B-Base                          | 69.00% | 100.00% |     69.00% | 138/200 | 200 | 0
Qwen/Qwen3-0.6B                               | 61.00% | 100.00% |     61.00% | 122/200 | 200 | 0
HuggingFaceTB/SmolLM2-360M                    | 52.00% |  86.00% |     60.47% | 104/200 | 172 | 28
nickypro/tinyllama-42M                        | 35.50% |  51.00% |     69.61% | 71/200 | 102 | 98
HuggingFaceTB/SmolLM2-135M                    | 33.50% |  49.50% |     67.68% | 67/200 | 99 | 101
Mostafa8Mehrabi/qwen3-30m-tinystories-final   | 30.00% |  45.00% |     66.67% | 60/200 | 90 | 110

=== Winogrande Accuracy Summary ===
Model                        | Acc% | Match% | Acc@Match% | Correct/Total | Matched | Unmatched
----------------------------------------------------------------------------------
HuggingFaceTB/SmolLM2-360M   | 35.20% |  71.70% |     49.09% | 352/1000 | 717 | 283
Qwen/Qwen3-0.6B-Base         | 32.00% |  61.90% |     51.70% | 320/1000 | 619 | 381
HuggingFaceTB/SmolLM2-1.7B   | 28.80% |  59.30% |     48.57% | 288/1000 | 593 | 407
HuggingFaceTB/SmolLM2-135M   | 26.80% |  55.20% |     48.55% | 268/1000 | 552 | 448
nickypro/tinyllama-42M       | 26.30% |  50.30% |     52.29% | 263/1000 | 503 | 497

=== SVAMP Accuracy Summary ===
Model                        | Acc% | Match% | Acc@Match% | Correct/Total | Matched | Unmatched
----------------------------------------------------------------------------------
Qwen/Qwen3-4B-base           | 41.50% | 100.00% |     41.50% | 83/200 | 200 | 0
HuggingFaceTB/SmolLM2-1.7B   | 16.00% | 100.00% |     16.00% | 32/200 | 200 | 0
Qwen/Qwen3-0.6B-Base         | 14.50% |  74.50% |     19.46% | 29/200 | 149 | 51
HuggingFaceTB/SmolLM2-360M   |  9.00% | 100.00% |      9.00% | 18/200 | 200 | 0
HuggingFaceTB/SmolLM2-135M   |  3.50% | 100.00% |      3.50% | 7/200 | 200 | 0
nickypro/tinyllama-42M       |  1.50% |  66.50% |      2.26% | 3/200 | 133 | 67
```