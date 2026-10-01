"""Routing: pluggable model selectors plus the lookup table that binds programs to them."""
from .base import ModelSelector
from .registry import available_selectors, make_selector, register_selector
from .bandit import LinUCB, LinTS, TypeUCB, EpsilonGreedy
from .static import Fixed, RoundRobin, Random, CallableSelector
from .acrouter import ACRouterSelector
from .routellm import RouteLLMSelector
from .features import FeatureExtractor, HashingFeatures, EmbeddingFeatures, TaskTypeFeatures, ConcatFeatures
from .reward import Reward
from .router import Router

__all__ = [
    "ModelSelector", "available_selectors", "make_selector", "register_selector",
    "LinUCB", "LinTS", "TypeUCB", "EpsilonGreedy", "Fixed", "RoundRobin", "Random", "CallableSelector",
    "ACRouterSelector", "RouteLLMSelector",
    "FeatureExtractor", "HashingFeatures", "EmbeddingFeatures", "TaskTypeFeatures", "ConcatFeatures",
    "Reward", "Router",
]
