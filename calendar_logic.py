from datetime import datetime

from logic import get_integrated_market_data


def get_today_economic_calendar():
    """経済指標カレンダー（Twelve Data）とニュースを取得"""
    data = get_integrated_market_data()
    return {
        "economic_calendar": data.get("economic_calendar", []),
        "news": data.get("news", []),
        "errors": data.get("errors", []),
    }


def get_today_market_events():
    """
    今日の為替市場で注目されている経済イベントをニュースから抽出
    """
    from newsapi import NewsApiClient
    import os
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return []
    newsapi = NewsApiClient(api_key=api_key)
    try:
        top_headlines = newsapi.get_everything(
            q='USD JPY 経済指標 OR 雇用統計 OR CPI',
            language='ja',
            sort_by='relevancy',
            page_size=5
        )

        events = []
        for article in top_headlines['articles']:
            events.append({
                "time": article['publishedAt'][:10], # 日付
                "event": article['title'],          # ニュースの見出し
                "source": article['source']['name'],
                "importance": "High"                 # ニュースになる＝重要度が高い
            })
        
        return events
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    print(get_today_market_events())