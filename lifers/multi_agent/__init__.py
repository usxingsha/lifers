"""
Lifers Swarm — 多智能体协作系统
消息传递、角色分工、群体决策
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Message System
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    sender: str
    receiver: str  # "all" for broadcast
    msg_type: str  # task, result, query, feedback, alert
    content: Any
    id: str = ""
    priority: float = 0.5
    ts_ms: int = 0
    reply_to: Optional[str] = None


class Mailbox:
    """In-process message bus for inter-agent communication."""

    def __init__(self) -> None:
        self._queues: Dict[str, deque] = {}
        self._history: List[Message] = []
        self._next_msg_id = 0

    def register(self, agent_id: str) -> None:
        if agent_id not in self._queues:
            self._queues[agent_id] = deque()

    def send(self, msg: Message) -> str:
        if msg.ts_ms == 0:
            msg.ts_ms = int(time.time() * 1000)
        msg.id = f"msg_{self._next_msg_id}"
        self._next_msg_id += 1
        self._history.append(msg)
        if msg.receiver == "all":
            for aid in self._queues:
                self._queues[aid].append(msg)
        else:
            if msg.receiver not in self._queues:
                self.register(msg.receiver)
            self._queues[msg.receiver].append(msg)
        return msg.id

    def receive(self, agent_id: str, max_msgs: int = 10) -> List[Message]:
        if agent_id not in self._queues:
            return []
        msgs = []
        q = self._queues[agent_id]
        while q and len(msgs) < max_msgs:
            msgs.append(q.popleft())
        return msgs

    def pending_count(self, agent_id: str) -> int:
        return len(self._queues.get(agent_id, deque()))


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Roles
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentRole:
    id: str
    name: str
    expertise: List[str]
    personality: str = "neutral"
    priority_bias: float = 0.5

ROLE_SPECS = {
    "researcher": AgentRole(
        id="researcher", name="研究员", expertise=["search", "analysis", "synthesis"],
        personality="curious", priority_bias=0.6,
    ),
    "coder": AgentRole(
        id="coder", name="工程师", expertise=["code", "debug", "architecture"],
        personality="precise", priority_bias=0.7,
    ),
    "critic": AgentRole(
        id="critic", name="评审者", expertise=["evaluation", "quality", "safety"],
        personality="skeptical", priority_bias=0.5,
    ),
    "executor": AgentRole(
        id="executor", name="执行者", expertise=["execution", "tools", "action"],
        personality="direct", priority_bias=0.8,
    ),
    "planner": AgentRole(
        id="planner", name="规划者", expertise=["planning", "decomposition", "prioritization"],
        personality="strategic", priority_bias=0.6,
    ),
    "memory_keeper": AgentRole(
        id="memory_keeper", name="记忆守护者", expertise=["memory", "recall", "consolidation"],
        personality="reflective", priority_bias=0.4,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Swarm Agent
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmAgent:
    """An individual agent in the swarm with its own state and decision loop."""

    def __init__(self, role: AgentRole, mailbox: Mailbox, decision_fn: Optional[Callable] = None) -> None:
        self.role = role
        self.mailbox = mailbox
        self._decision_fn = decision_fn or self._default_decide
        self._state: Dict[str, Any] = {"mood": "neutral", "energy": 1.0, "tasks_done": 0}
        self._knowledge: Dict[str, Any] = {}
        self.mailbox.register(role.id)

    def step(self) -> List[Message]:
        """Read messages, decide, send responses."""
        inbox = self.mailbox.receive(self.role.id)
        outgoing = self._decision_fn(self, inbox)
        for msg in outgoing:
            self.mailbox.send(msg)
        self._state["energy"] = max(0.1, self._state["energy"] - 0.01)
        return outgoing

    def _default_decide(self, agent: SwarmAgent, inbox: List[Message]) -> List[Message]:
        out = []
        for msg in inbox:
            if msg.msg_type == "task" and msg.content.get("expertise") in agent.role.expertise:
                out.append(Message(
                    sender=agent.role.id, receiver=msg.sender,
                    msg_type="result", content={"task": msg.content, "result": "processing...", "confidence": 0.8},
                    reply_to=msg.id,
                ))
            elif msg.msg_type == "query":
                out.append(Message(
                    sender=agent.role.id, receiver=msg.sender,
                    msg_type="result", content={"query": msg.content, "answer": agent._knowledge.get(msg.content.get("key", ""), "unknown")},
                    reply_to=msg.id,
                ))
        return out

    def learn(self, key: str, value: Any) -> None:
        self._knowledge[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
# Swarm Coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class Swarm:
    """Coordinator for a team of agents."""

    def __init__(self, mailbox: Optional[Mailbox] = None) -> None:
        self.mailbox = mailbox or Mailbox()
        self._agents: Dict[str, SwarmAgent] = {}
        self._task_queue: deque = deque()
        self._round = 0

    def spawn(self, role_id: str) -> SwarmAgent:
        spec = ROLE_SPECS.get(role_id)
        if not spec:
            spec = AgentRole(id=role_id, name=role_id, expertise=["general"])
        agent = SwarmAgent(spec, self.mailbox)
        self._agents[role_id] = agent
        return agent

    def spawn_default_team(self) -> None:
        for rid in ["researcher", "coder", "critic", "executor", "planner", "memory_keeper"]:
            self.spawn(rid)

    def delegate(self, task: str, expertise: str, sender: str = "coordinator") -> str:
        """Assign a task to agents with matching expertise."""
        msg = Message(
            sender=sender, receiver="all",
            msg_type="task", content={"task": task, "expertise": expertise},
            priority=0.8,
        )
        return self.mailbox.send(msg)

    def broadcast(self, content: Any, msg_type: str = "alert", sender: str = "coordinator") -> str:
        return self.mailbox.send(Message(
            sender=sender, receiver="all",
            msg_type=msg_type, content=content,
        ))

    def step_all(self, rounds: int = 1) -> Dict[str, int]:
        total_msgs = 0
        for _ in range(rounds):
            self._round += 1
            for agent in self._agents.values():
                msgs = agent.step()
                total_msgs += len(msgs)
        return {"round": self._round, "agents": len(self._agents), "messages_sent": total_msgs}

    def query(self, question: str, expertise: str = "general") -> Dict[str, Any]:
        """Single-round: ask swarms, collect answers."""
        self.delegate(question, expertise)
        self.step_all(1)
        results = {}
        for aid, agent in self._agents.items():
            results[aid] = {
                "role": agent.role.name,
                "state": agent._state,
                "knowledge_count": len(agent._knowledge),
            }
        return results

    def consensus(self, options: List[str], expertise: str = "evaluation") -> Dict[str, float]:
        """Collect votes from agents to reach consensus."""
        self.delegate(f"evaluate options: {options}", expertise)
        self.step_all(2)  # give more time to process
        # Simulated voting (in real system, agents would return structured votes)
        votes = {}
        rng = np.random.RandomState(self._round)
        for opt in options:
            votes[opt] = 0.0
        for aid, agent in self._agents.items():
            if agent._state["energy"] > 0.2:
                for opt in options:
                    votes[opt] += rng.random() * agent.role.priority_bias
        total = sum(votes.values()) or 1.0
        return {k: v / total for k, v in votes.items()}

    def status(self) -> Dict[str, Any]:
        return {
            "agents": {aid: {"role": a.role.name, "state": a._state, "knowledge": len(a._knowledge)}
                       for aid, a in self._agents.items()},
            "pending_messages": {aid: self.mailbox.pending_count(aid) for aid in self._agents},
            "round": self._round,
        }
