import pandas as pd
import json

instruction = """你是擅长使用SQL帮助用户进行数据分析的智能助手，尽你所能的根据下面的用户问题生成准确的SQL语句。

这里是数据库表的结构信息：
CREATE TABLE IF NOT EXISTS dwd_en_pub_customs_trade_hscode_df_lm (
year STRING COMMENT '年份',
 month STRING COMMENT '月份',
 direction STRING COMMENT '贸易类型：Import/Export',
 hscode_type STRING COMMENT '海关商品编码类型',
 hscode STRING COMMENT '海关商品编码',
 reporter_country_en_name STRING COMMENT '报告国家英文名（IHS提供的）',
 reporter_country_id STRING COMMENT '报告国家ID',
 reporter_country_iso STRING COMMENT '报告国家国际标准ISO（IHS提供的）',
 partner_country_en_name STRING COMMENT '贸易伙伴国家英文名（IHS提供的）',
 partner_country_id STRING COMMENT '贸易伙伴国家ID',
 partner_country_iso STRING COMMENT '贸易伙伴国家国际标准ISO（ISH提供的）',
 currency STRING COMMENT '币种，当前只有USD',
 trade_amt DOUBLE COMMENT '贸易金额',
 ds STRING COMMENT  '分区字段，仅保留最新分区'
)
COMMENT '该数据表包含了40个核心国家的进出口贸易数据，用于内外部数据产品市场趋势分析'
LIFECYCLE 720;

这里数据表中的几条样例数据：
[['year', 'month', 'direction', 'hscode_type', 'hscode', 'reporter_country_en_name', 'reporter_country_id', 'reporter_country_iso', 'partner_country_en_name', 'partner_country_id', 'partner_country_iso', 'currency', 'trade_amt', 'ds'], ['2020', '05', 'Export', 'hs6', '020230', 'Russia', 'RU', 'RU', 'Armenia', 'Notcare', 'AM', 'USD', '36879.65', '20240128'], ['2022', '01', 'Export', 'hs6', '020329', 'Russia', 'RU', 'RU', 'Armenia', 'Notcare', 'AM', 'USD', '63165.78', '20240128'], ['2023', '02', 'Import', 'hs6', '020443', 'Russia', 'RU', 'RU', 'Belarus', 'Notcare', 'BY', 'USD', '2546.65', '20240128'], ['2020', '07', 'Export', 'hs6', '020649', 'Russia', 'RU', 'RU', 'Hong Kong', 'HK', 'HK', 'USD', '2452103.62', '20240128'], ['2021', '09', 'Import', 'hs6', '020714', 'Russia', 'RU', 'RU', 'Korea, North', 'KP', 'KP', 'USD', '52459.0', '20240128']]

这里是一些相似的用户问题和对应的SQL语句：
[{'input': '2020年1月KH从FR进口电动真空吸尘器功率≤1500W集尘器容积≤20L的金额（hscode主要包括850811）', 'query': "SELECT SUM(trade_amt) AS total_trade_amt FROM dwd_en_pub_customs_trade_hscode_df_lm WHERE year = '2020' AND month = '01' AND direction = 'Import' AND reporter_country_en_name = 'Cambodia' AND partner_country_en_name = 'France' AND hscode = '850811' AND ds = '20220102';"}, {'input': '能否提供2021年上半年的进口到Japan的金额？', 'query': "SELECT SUM(trade_amt) AS total_trade_amt FROM dwd_en_pub_customs_trade_hscode_df_lm WHERE ds = '20230815' AND direction = 'Import' AND reporter_country_en_name='Japan' AND year = '2021' AND month >= '01' AND month < '07';"}, {'input': '你知不知道近两年埃及进口的182132的最多月份是几月', 'query': "SELECT year, month, SUM(trade_amt) AS total_trade_amt FROM dwd_en_pub_customs_trade_hscode_df_lm WHERE reporter_country_en_name = 'Egypt' AND hscode = '182132' AND direction = 'Import' AND ds = '20220701' AND CONCAT(year, month) >= TO_CHAR(DATEADD(TO_DATE(ds, 'yyyymmdd'), _24, 'mm'), 'yyyymm') AND CONCAT(year, month) < TO_CHAR(TO_DATE(ds, 'yyyymmdd'), 'yyyymm') GROUP BY year, month ORDER BY SUM(trade_amt) DESC LIMIT 1;"}]

这里与用户问题相关的领域知识：
['增长最快是指当前周期内的总金额SUM(trade_amt)环比上一个周期内总金额的增长比例最高', '趋势是指每个月的金额情况，需要年（year）、month(月)、sum(trade_amt)(总金额)字段，需要按照时间远到近排序，比如ORDER BY year, month ASC', '增幅是指当前周期内的总金额SUM(trade_amt)环比上一个周期内总金额的增长比例', '旺季月份是指销售总金额SUM(trade_amt)最多的月份', 'Import表示进口，Export表示出口']

请使用下面的格式输出，且请勿输出任务其它信息：
Thought: 你在这里分析用户问题能否如何转成SQL语句，以及将用户问题转成SQL语句的具体思路

```sql
你在这里将用户问题转成SQL语句
```


开始！

"""

query_sql= pd.read_excel("./customs_text2sql.xlsx")

samples = []
with open("./customs_gold.txt", "w") as w:
    for query, sql in query_sql.values:
        query = query.replace("\n", " ")
        sql = sql.replace("\n", " ").replace("  ", " ").replace("  ", " ").replace(";", "")
        w.write(sql + " customs\n")
        sample = {
            "db_id": "customs",
            "instruction": instruction,
            "input": "用户问题：" + query,
            "output": sql,
            "history": []
        }
        samples.append(sample)

with open("./customs_text2sql_dev.json", 'w') as w:
    w.write(json.dumps(samples, ensure_ascii=False))