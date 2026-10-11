"""集中設定。應用程式應經由 get_settings() 取得這次 run 的參數。"""

from config.settings import Settings, ensure_python, get_settings

__all__ = ["Settings", "ensure_python", "get_settings"]
