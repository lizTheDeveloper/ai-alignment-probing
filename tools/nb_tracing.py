"""Shared Langfuse tracing cell for the notebook builders.

The generated cell reads keys from a .env file next to the notebook (never from the notebook itself),
so notebooks can be published without secrets. With no keys, tracing silently turns off.
"""


def tracing_cell(notebook_name):
    return '''
# --- Langfuse tracing (optional) ---
# Put LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL in a .env file next to this notebook.
# Optionally set STUDENT_ID in .env so your runs are grouped under your name.
# With no keys, everything below still runs; it just isn't traced.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "langfuse>=4", "python-dotenv"])

import os
from contextlib import contextmanager
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
NOTEBOOK = "NOTEBOOK_NAME"
SESSION_ID = f"{NOTEBOOK}-{datetime.now():%Y%m%d-%H%M%S}"

class _NoTrace:
    def update(self, **kwargs): pass
    def score(self, **kwargs): pass

TRACING = False
if os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client, propagate_attributes
    langfuse = get_client()
    TRACING = langfuse.auth_check()

@contextmanager
def trace(name, as_type="span", **kwargs):
    """A Langfuse observation (span or generation) grouped under this notebook run. No-op without keys."""
    if not TRACING:
        yield _NoTrace()
        return
    with propagate_attributes(session_id=SESSION_ID, user_id=os.environ.get("STUDENT_ID"),
                              tags=["ai-alignment", NOTEBOOK]):
        with langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
            yield obs

print(f"Langfuse tracing: {'on, session ' + SESSION_ID if TRACING else 'off (no keys in .env)'}")
'''.replace("NOTEBOOK_NAME", notebook_name)


FLUSH_CELL = '''
# Send any buffered traces to Langfuse
if TRACING:
    langfuse.flush()
    print(f"Traces flushed. Find this run in Langfuse under session {SESSION_ID}")
'''
