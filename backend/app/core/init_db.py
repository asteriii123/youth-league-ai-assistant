"""
数据库初始化模块。

负责：

1. 创建数据表
2. 初始化数据库环境


项目第一次运行时执行。
"""
from sqlalchemy import inspect, select, text

from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.models.entities import ClassRoom, MeetingJob, User


DEMO_CLASSES = [
    ("23级计算机科学与技术1班", "JSJ23-1"),
    ("23级软件工程1班", "RJGC23-1"),
    ("23级数据科学与大数据技术1班", "SJ23-1"),
]


def apply_local_migrations() -> None:
    """Apply small, data-preserving SQLite upgrades for the local application."""
    inspector = inspect(engine)
    if "notices" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("notices")}
        if "attachment_name" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE notices ADD COLUMN attachment_name VARCHAR(255)")
                )

    inspector = inspect(engine)
    if "collection_tasks" not in inspector.get_table_names():
        return
    task_columns = {column["name"] for column in inspector.get_columns("collection_tasks")}
    statements: list[str] = []
    if "status" not in task_columns:
        statements.append("ALTER TABLE collection_tasks ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'draft'")
    if "attachment_required" not in task_columns:
        statements.append("ALTER TABLE collection_tasks ADD COLUMN attachment_required BOOLEAN NOT NULL DEFAULT 0")
    if "allow_modify" not in task_columns:
        statements.append("ALTER TABLE collection_tasks ADD COLUMN allow_modify BOOLEAN NOT NULL DEFAULT 1")
    if "updated_at" not in task_columns:
        statements.append("ALTER TABLE collection_tasks ADD COLUMN updated_at DATETIME")
    if statements:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            if "requires_file" in task_columns:
                connection.execute(text("UPDATE collection_tasks SET attachment_required = requires_file"))
            if "allow_update" in task_columns:
                connection.execute(text("UPDATE collection_tasks SET allow_modify = allow_update"))
            connection.execute(text("UPDATE collection_tasks SET updated_at = created_at WHERE updated_at IS NULL"))

    inspector = inspect(engine)
    if "meeting_records" not in inspector.get_table_names():
        return
    meeting_columns = {column["name"] for column in inspector.get_columns("meeting_records")}
    meeting_statements: list[str] = []
    if "summary" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
    if "key_points_json" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN key_points_json TEXT NOT NULL DEFAULT '[]'")
    if "decisions_json" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN decisions_json TEXT NOT NULL DEFAULT '[]'")
    if "action_items_json" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN action_items_json TEXT NOT NULL DEFAULT '[]'")
    if "source_path" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN source_path TEXT")
    if "source_name" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN source_name VARCHAR(255)")
    if "updated_at" not in meeting_columns:
        meeting_statements.append("ALTER TABLE meeting_records ADD COLUMN updated_at DATETIME")
    if meeting_statements:
        with engine.begin() as connection:
            for statement in meeting_statements:
                connection.execute(text(statement))
            if "summary_json" in meeting_columns:
                connection.execute(text("UPDATE meeting_records SET summary = summary_json WHERE summary = ''"))
            connection.execute(text("UPDATE meeting_records SET updated_at = created_at WHERE updated_at IS NULL"))

    inspector = inspect(engine)
    if "knowledge_documents" not in inspector.get_table_names():
        return
    knowledge_columns = {column["name"] for column in inspector.get_columns("knowledge_documents")}
    knowledge_statements: list[str] = []
    if "index_status" not in knowledge_columns:
        knowledge_statements.append("ALTER TABLE knowledge_documents ADD COLUMN index_status VARCHAR(20) NOT NULL DEFAULT 'pending'")
    if "index_error" not in knowledge_columns:
        knowledge_statements.append("ALTER TABLE knowledge_documents ADD COLUMN index_error TEXT")
    if "indexed_at" not in knowledge_columns:
        knowledge_statements.append("ALTER TABLE knowledge_documents ADD COLUMN indexed_at DATETIME")
    if knowledge_statements:
        with engine.begin() as connection:
            for statement in knowledge_statements:
                connection.execute(text(statement))

    inspector = inspect(engine)
    if "meeting_jobs" in inspector.get_table_names():
        job_columns = {column["name"] for column in inspector.get_columns("meeting_jobs")}
        if "instruction" not in job_columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE meeting_jobs ADD COLUMN instruction TEXT NOT NULL DEFAULT '请整理为标准会议纪要'"))


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    apply_local_migrations()
    with SessionLocal() as db:
        db.query(MeetingJob).filter(MeetingJob.status.in_([
            "queued", "transcribing", "filtering", "generating_minutes", "creating_document"
        ])).update({"status": "failed", "error_message": "后端曾在处理期间停止，请点击重试"}, synchronize_session=False)
        for index, (name, invite_code) in enumerate(DEMO_CLASSES, start=1):
            classroom = db.scalar(select(ClassRoom).where(ClassRoom.name == name))
            if not classroom:
                classroom = ClassRoom(name=name, invite_code=invite_code)
                db.add(classroom)
                db.flush()
            username = f"secretary{index}"
            if not db.scalar(select(User).where(User.username == username)):
                db.add(User(
                    username=username,
                    password_hash=hash_password("123456"),
                    display_name=f"{name}团支书",
                    role="secretary",
                    class_id=classroom.id,
                ))
        db.commit()


if __name__ == "__main__":
    initialize_database()
    print("本地数据库初始化完成。演示团支书账号：secretary1 / 123456")
