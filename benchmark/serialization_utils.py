"""Utilities for JSON serialization of benchmark results."""
from typing import Any, Dict, List


def sanitize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove non-JSON-serializable objects (tensors) from tool call results.
    
    Args:
        tool_calls: List of tool call dicts potentially containing tensors
        
    Returns:
        List of sanitized tool call dicts safe for JSON serialization
    """
    sanitized = []
    for tc in tool_calls:
        clean_tc = {
            "tool_name": tc.get("tool_name"),
            "arguments": tc.get("arguments"),
        }
        # Only keep the text part of results, drop vision_features (tensors)
        result = tc.get("result", {})
        if isinstance(result, dict):
            clean_tc["result"] = {"text": result.get("text", "")}
        else:
            clean_tc["result"] = str(result)
        sanitized.append(clean_tc)
    return sanitized

