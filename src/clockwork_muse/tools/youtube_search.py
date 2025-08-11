import os, requests

class YouTubeSearchTool:
    """Simple YouTube Data API v3 search tool.
    Usage: tool.run({"q": "query", "max_results": 5}) -> list of dicts
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not set")

    def run(self, params: dict) -> list:
        q = params.get("q", "")
        max_results = int(params.get("max_results", 5))
        url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&type=video&maxResults={max_results}&q={requests.utils.quote(q)}&key={self.api_key}"
        )
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("items", []):
            vid = item["id"]["videoId"]
            sn = item["snippet"]
            out.append({
                "videoId": vid,
                "title": sn.get("title"),
                "channel": sn.get("channelTitle"),
                "publishedAt": sn.get("publishedAt"),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
        return out