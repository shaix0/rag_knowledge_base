MODE_CONFIG = {
    # "selection": {
    #     "title": "選擇測驗模式",
    #     "icon": "🩺",
    #     "description": "請選擇一種測驗模式來開始練習。",
    #     "limit": 0,
    #     "time": 0
    # },
    "normal": {
        "title": "標準模式",
        "icon": "🎲",
        "description": "從所有題庫中隨機選取題目，進行全面測試。",
        "limit": 50, # 題目數量上限
        "time": 100  # 分鐘
    },
    "practice": {
        "title": "錯題重練模式",
        "icon": "🔁",
        "description": "僅包含過去答錯或標記收藏的題目，鞏固學習。",
        "limit": 50,
        "time": 100
    },
    "weakness": {
        "title": "弱點加強模式",
        "icon": "💪",
        "description": "優先練習錯誤率較高的題目，提升整體表現。",
        "limit": 50,
        "time": 100
    }
}