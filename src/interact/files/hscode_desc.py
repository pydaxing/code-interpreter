import pandas as pd
import os
import json
import odps



# o = odps.ODPS(access_id="xxxx",
#               secret_access_key="xxxxxx",
#               project="xxxx",
#               endpoint="xxxx")
#
#
# table = o.get_table("tars_agent_hscode_nl2sql_description")
# global hscode_desc
# with table.open_reader(partition="ds=20231129") as reader:
#     hscode_desc = reader.to_pandas()
#
# hscode_desc.to_excel("../raw/海关hscode描述.xlsx")



prompt_template = """# TASK
你的任务是将关于海关数据的 query 总结成如下模板，要求尽量简洁。
提供一个改写版本，要求表达意图一致，并且需要在表达层面进行 paraphrasing。
并且提供一个复杂版本，这个版本可能需要涉及到 query 背后用户的真实世界的情况，以及他对于海关数据查询及分析的动机所在，我希望它包含更加丰富的内容。
所有版本均要求：保证原始 query 所有信息都被保留，并且不能改动或增加额外的海关数据查询/分析的需求。

# Template
“{时间/日期} {出口国} 向 {进口国} 出口 {商品名称/编码} 的 {问题类型}”
“{时间/日期} {进口国} 从 {出口国} 进口 {商品名称/编码} 的 {问题类型}”

其中：
- 时间/日期可以是具体的年份、月份、季度或时间段，如"2022年1月"、“近半年"等。
- 出口国和进口国是指定的国家，如"中国”、“美国"等。
- 商品名称/编码可以是具体的商品名称或海关编码，如"陶瓷制绝缘零件”、“品目7321"等。
- 问题类型可以是"金额”、“排名”、“趋势"、“增长最快的国家"、“峰谷月份"、“海关数据分析”等。

# Example
## Input
“请帮我查询一下从荷兰进口的其他纺织材料制刺绣品，镍丝布、格栅及网和栎木（橡木）原木的最高6个月份，时间范围限定在2022年上半年，然后告诉我”
## Output
```
{
    "rationale": "用户想要查询关于从荷兰进口到其他国家特定品类下的海关数据，候选列表中#API_NAME#能满足此需求，我将直接调用此工具。",
    "summarized_version":"2022年上半年，所有国家从荷兰进口的其他纺织材料制刺绣品，镍丝布、格栅及网和栎木橡木原木，总量最高6个月份", 
    "alter_verison":"所有国家从NL，针对于其他纺织材料制刺绣品，镍丝布、格栅及网和栎木（橡木）原木的进口量，在2022年上半年中最高6个月份是哪些啊，你能否帮我查一下",
    "complex_version":"我是一家位于CN的进口商，我们主要从Netherlands进口其他纺织材料制刺绣品，镍丝布、格栅及网和栎木（橡木）原木。由于我们正在进行年度财务审计，我需要了解在2022年上半年中，所有国家从Netherlands进口这些商品的数据情况。为了更好地理解我们的进口情况和进行未来的市场预测，我特别关注进口金额最高的6个月份。能否帮我查询这些信息？"
}
```

## Input
“我这边急等着做数据盘点，能否尽快给我提供一下尼日利亚进口到意大利的金额？提供2020年下半年的就行”
{
    "rationale": "用户意图为查询尼日利亚进口到意大利的金额总量且不限商品品类，候选工具#API_NAME#可应对海关数据分析问题，我将使用此工具。",
    "summarized_version": "2020年下半年，意大利从尼日利亚进口的全部商品的金额",
    "alter_verison": "意大利从NG，所有商品的进口金额，时间范围在2020年下半年",
    "complex_version": "我是一家经营在Italy的进口公司，我们主要从Nigeria进口商品。为了更好地了解我所在的市场，我非常希望了解2020年下半年，我们在这个时间段内，所进口的商品的总金额是多少。快给我查出来！"
}

# Exception
用户可能存在不标准的表达（如，XX进口到YY，XX向YY进口，XX从YY出口），但我要求你，在 summarized_version 中必须归一表达为 “XX 从 YY 进口”或“XX 出口到 YY” 两种表达之一。如果缺乏进出口的主体或客体，请填充“所有国家”。

A从B进口 -> A 从 B 进口
A到B进口 -> A 从 B 进口
从A进口到B -> B 从 A 进口
从A进口 = 所有国家 从 A 进口
A进口 -> A 从 所有国家 进口

A向B出口 -> A 出口到 B
从A到B出口 -> A 出口到 B
从A出口到B -> A 出口到 B
向A出口 -> 所有国家 出口到 A
出口到A出口 -> 所有国家 出口到 A

# Output Format
- Return a valid JSON with keys “rationale”, “summarized_version”, “alter_verison” and “complex_version”, its value being a list of dictionaries. Enclose with three backticks.
- 要求：“alter_verison” and “complex_version” 中的"国家名称"的表达不能是中文，需要使用英文（America、Britain、India等）或Country Codes（如 US、UK、IN等）来描述国家，改写 query 的内容仍需要是中文。而 “summarized_version” 中的国家名称必须是中文。
- 对 “summarized_version” 要求：不能表达为 “XX进口到YY”，“XX向YY进口”，“XX从YY出口”。必须归一表达为 “XX 从 YY 进口”或“XX 出口到 YY” 两种表达之一。
- 要求 “rationale” 尽量简洁、准确，结构上包含“用户意图分析-工具调用决策”。

# Input
“#QUERY#”
# Output
"""

hscode_desc = pd.read_excel("../raw/海关hscode描述.xlsx")

print(hscode_desc)