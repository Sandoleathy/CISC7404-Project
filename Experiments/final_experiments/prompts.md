```python
DA_PREAMBLE = ("Answer with the final answer only: no calculations, no pointer steps, no intermediate "
               "values, no explanation and no comment. Write only the lines that the rules below require, "
               "plus the final answer line. Any other text makes the response invalid.\n")

BASE_RULE = ("Output format. Write the lines required by the rules below first, in the order the rules are "
             "listed, and then write the final answer line. The answer line is always the very last line of "
             "the response: ####ANSWER= followed by the entry label as one letter, and nothing may be "
             "written after it. Do not repeat or restate these rules. Do not write any keyword line that "
             "these rules do not require.")

COT_PREAMBLE = ("Solve the problem step by step. Before the final answer line, explicitly write the entry "
                "reached after each pointer step, one per line. The keyword lines required by the rules "
                "below must come after those steps and before the final answer line.\n")
```