import sys
import os

sys.path.append(os.getcwd())
# 챗봇 데이터 학습용 테이블 생성
import pymysql
from config.DatabaseConfig import *

db = None

try:
    db = pymysql.connect(
        host = DB_HOST,
        user = DB_USER,
        passwd = DB_PASSWARD,
        db = DB_NAME,
        charset = 'utf8'
    )

    # 테이블 생성 sql 정의
    
    sql = '''
    CREATE TABLE if not exists `chatbot_train_data` (
        `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
        `intent` VARCHAR(45) NULL,
        `ner` VARCHAR(1024) NULL,
        `query` TEXT NULL,
        `answer` TEXT NOT NULL,
        `answer_image` TEXT NULL,
        PRIMARY KEY (`id`))
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = utf8;

    '''

    # 테이블 생성
    with db.cursor() as cursor:
        cursor.execute(sql)
except Exception as e:
    print(e)
finally:
    if db is not None:
        db.close()