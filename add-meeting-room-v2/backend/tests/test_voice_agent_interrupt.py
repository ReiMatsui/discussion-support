import asyncio

from app.core.config import Settings
from app.voice_agent.orchestrator import VoiceAgentOrchestrator


class FakePlayer:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


def test_interrupt_aborts_audio_and_cancels_response_task() -> None:
    async def scenario() -> None:
        orchestrator = VoiceAgentOrchestrator(
            settings=Settings(
                openai_api_key="test",
                voice_agent_enabled=True,
            )
        )
        player = FakePlayer()
        response_task = asyncio.create_task(asyncio.sleep(60))
        orchestrator._agent_speaking = True
        orchestrator._current_player = player
        orchestrator._pending_task = response_task

        interrupted = await orchestrator.interrupt(reason="test_human_speech")

        assert interrupted is True
        assert player.aborted is True
        assert response_task.cancelled()
        assert orchestrator.is_speaking is False
        assert orchestrator._last_agent_finished_ms == 0
        assert orchestrator._interrupted is False

    asyncio.run(scenario())


def test_interrupt_does_nothing_when_agent_is_not_speaking() -> None:
    async def scenario() -> None:
        orchestrator = VoiceAgentOrchestrator(
            settings=Settings(
                openai_api_key="test",
                voice_agent_enabled=True,
            )
        )

        interrupted = await orchestrator.interrupt(reason="test_human_speech")

        assert interrupted is False

    asyncio.run(scenario())
