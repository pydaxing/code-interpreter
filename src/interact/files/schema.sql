--
-- SQLiteStudio v3.4.4 生成的文件，Wed Feb 21 11:43:21 2024
--
-- 所用的文本编码：UTF-8
--
PRAGMA foreign_keys = off;
BEGIN TRANSACTION;

-- 表：hi
DROP TABLE IF EXISTS hi;

CREATE TABLE IF NOT EXISTS hi (
    id   TEXT    PRIMARY KEY
                 NOT NULL
                 UNIQUE,
    name TEXT,
    age  NUMERIC
);


COMMIT TRANSACTION;
PRAGMA foreign_keys = on;
