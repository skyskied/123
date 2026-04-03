import mysql.connector
from mysql.connector import errorcode
import bcrypt

DB_CONFIG = {
    'user': 'root',
    'password': 'hjj040201',
    'host': 'localhost',
    'database': 'users_data',
}

def create_database_if_not_exists():
    """如果数据库不存在，则创建数据库"""
    try:
        # 先不指定数据库进行连接
        cnx = mysql.connector.connect(user=DB_CONFIG['user'], password=DB_CONFIG['password'], host=DB_CONFIG['host'])
        cursor = cnx.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} DEFAULT CHARACTER SET 'utf8'")
        print(f"Database '{DB_CONFIG['database']}' created or already exists.")
        cursor.close()
        cnx.close()
    except mysql.connector.Error as err:
        print(f"Failed to create database: {err}")
        exit(1)

def get_db_connection():
    """获取数据库连接"""
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("用户名或密码错误")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("数据库不存在")
        else:
            print(err)
        return None

def create_users_table():
    """创建 users 表"""
    cnx = get_db_connection()
    if not cnx:
        return

    cursor = cnx.cursor()
    table_description = "".join((
        "CREATE TABLE IF NOT EXISTS `users` ( ",
        "`id` INT AUTO_INCREMENT PRIMARY KEY, ",
        "`username` VARCHAR(50) NOT NULL UNIQUE, ",
        "`password` VARCHAR(255) NOT NULL, ",
        "`email` VARCHAR(100) NOT NULL UNIQUE, ",
        "`is_admin` TINYINT(1) NOT NULL DEFAULT 0, ",
        "`avatar` VARCHAR(255) DEFAULT NULL, ",
        "`created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        ") ENGINE=InnoDB"))
    try:
        print("正在创建表 'users': ")
        cursor.execute(table_description)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("表已存在。")
        else:
            print(err.msg)
    else:
        print("OK")
    cursor.close()
    cnx.close()

def add_user(username, password, email, is_admin=0):
    """添加新用户，对密码进行哈希处理"""
    cnx = get_db_connection()
    if not cnx:
        return False, "数据库连接失败"

    cursor = cnx.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        add_user_query = ("INSERT INTO users "
                          "(username, password, email, is_admin) "
                          "VALUES (%s, %s, %s, %s)")
        user_data = (username, hashed_password, email, is_admin)
        cursor.execute(add_user_query, user_data)
        cnx.commit()
        return True, "用户注册成功"
    except mysql.connector.Error as err:
        if err.errno == 1062:  # Duplicate entry
            return False, "用户名或邮箱已被注册"
        else:
            return False, f"注册失败: {err}"
    finally:
        cursor.close()
        cnx.close()

def verify_user(username, password):
    """验证用户名和密码"""
    cnx = get_db_connection()
    if not cnx:
        return False, "数据库连接失败", None

    cursor = cnx.cursor(dictionary=True)
    try:
        query = "SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return True, "登录成功", user  # Return full user object/dict
        else:
            return False, "用户名或密码错误", None
    except mysql.connector.Error as err:
        return False, f"登录失败: {err}", None
    finally:
        cursor.close()
        cnx.close()

def get_all_users():
    """获取所有用户列表 (管理员用)"""
    cnx = get_db_connection()
    if not cnx:
        return []

    cursor = cnx.cursor(dictionary=True)
    try:
        query = "SELECT id, username, email, is_admin, avatar, created_at FROM users"
        cursor.execute(query)
        users = cursor.fetchall()
        return users
    except mysql.connector.Error as err:
        print(f"获取用户列表失败: {err}")
        return []
    finally:
        cursor.close()
        cnx.close()

def delete_user(user_id):
    """删除用户 (管理员用)"""
    cnx = get_db_connection()
    if not cnx:
        return False, "数据库连接失败"

    cursor = cnx.cursor()
    try:
        query = "DELETE FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        cnx.commit()
        return True, "用户删除成功"
    except mysql.connector.Error as err:
        return False, f"删除失败: {err}"
    finally:
        cursor.close()
        cnx.close()

def get_user_by_id(user_id):
    """通过ID获取用户信息"""
    cnx = get_db_connection()
    if not cnx:
        return None

    cursor = cnx.cursor(dictionary=True)
    try:
        query = "SELECT * FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        return user
    except mysql.connector.Error as err:
        print(f"查询用户失败: {err}")
        return None
    finally:
        cursor.close()
        cnx.close()

def update_user_info(user_id, username, email, password=None, avatar=None):
    """更新用户信息"""
    cnx = get_db_connection()
    if not cnx:
        return False, "数据库连接失败"

    cursor = cnx.cursor()
    try:
        # 构建动态更新语句
        updates = []
        params = []

        if username:
            updates.append("username = %s")
            params.append(username)
        if email:
            updates.append("email = %s")
            params.append(email)
        if password:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            updates.append("password = %s")
            params.append(hashed_password)
        if avatar:
            updates.append("avatar = %s")
            params.append(avatar)
            
        if not updates:
            return False, "没有需要更新的信息"

        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
        
        cursor.execute(query, tuple(params))
        cnx.commit()
        return True, "信息更新成功"
    except mysql.connector.Error as err:
        return False, f"更新失败: {err}"
    finally:
        cursor.close()
        cnx.close()

def get_user_by_email(email):
    """通过邮箱获取用户信息"""
    cnx = get_db_connection()
    if not cnx:
        return None

    cursor = cnx.cursor(dictionary=True)
    try:
        query = "SELECT * FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        return user
    except mysql.connector.Error as err:
        print(f"查询用户失败: {err}")
        return None
    finally:
        cursor.close()
        cnx.close()

def update_password(email, new_password):
    """更新用户密码"""
    cnx = get_db_connection()
    if not cnx:
        return False, "数据库连接失败"

    cursor = cnx.cursor()
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    try:
        query = "UPDATE users SET password = %s WHERE email = %s"
        cursor.execute(query, (hashed_password, email))
        cnx.commit()
        return True, "密码更新成功"
    except mysql.connector.Error as err:
        return False, f"密码更新失败: {err}"
    finally:
        cursor.close()
        cnx.close()

if __name__ == '__main__':
    create_database_if_not_exists()
    create_users_table()
