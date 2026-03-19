Input  : "COVID vaccines cause infertility and are banned in Europe"
Process: LLM breaks it into atomic verifiable sub-claims
Output : [
    "COVID vaccines cause infertility",
    "COVID vaccines are banned in Europe"
]

Why atomic?
One claim = one retrieval = one verdict
Compound claims confuse retrieval — you get mixed evidence