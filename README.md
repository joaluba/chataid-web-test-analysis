# chataid-web-test-analysis

The following analysis is performed:

### SUBJECTIVE RESULTS (human ratings):

Reading, plotting and analysis of human ratings. 

### OBJECTIVE RESULTS:

1. Deriving transcript from audio recordings.
2. Scoring the information collection task (to get Information Collection Score).
3. Counting the number of clarification requests (to get Number Of Clarification Request).

Some thoughts on automating the analysis: In general using Gemini API for data analysis appears so easy and practical that it is very tempting to just rely on it... But using LLM as a scoring system is so far not reliable or reproducible: we cannot guarantee that the model will be deterministic (even when using the temperature of 0)