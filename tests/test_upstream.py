import unittest
import json

from genai_proxy.tools import parse_tool_calls_from_content
from genai_proxy.upstream import iter_sse_events


class UpstreamSseTests(unittest.TestCase):
    def test_last_delta_is_kept_when_finish_reason_is_present(self):
        fragments = ['{"tool_calls":[', '{"name":"weather","arguments":{}}]}']
        lines = [
            'data: ' + json.dumps({"choices": [{
                "delta": {"content": fragment},
                "finish_reason": "stop" if index else None,
            }]})
            for index, fragment in enumerate(fragments)
        ]

        events = list(iter_sse_events(lines))
        content = "".join(
            event.get("content") or ""
            for event in events
            if event["type"] == "delta"
        )

        self.assertEqual(content, "".join(fragments))
        self.assertEqual(parse_tool_calls_from_content(content)[0]["function"]["name"], "weather")
        self.assertEqual(events[-1]["type"], "done")

    def test_business_error_is_not_an_empty_success(self):
        for payload in [
            {"code": 500, "errMsg": "未找到模型"},
            {"success": False, "message": "Token失效，请重新登录"},
        ]:
            with self.subTest(payload=payload):
                events = list(iter_sse_events(['data: ' + json.dumps(payload)]))
                self.assertEqual([event["type"] for event in events], ["error"])
                self.assertEqual(events[0]["error"], payload.get("errMsg") or payload["message"])

    def test_tool_call_json_is_parsed_after_stream_aggregation(self):
        content = '{"tool_calls":[{"name":"weather","arguments":{"city":"上海"}}]}'

        calls = parse_tool_calls_from_content(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(calls[0]["function"]["arguments"], '{"city": "上海"}')


if __name__ == "__main__":
    unittest.main()
