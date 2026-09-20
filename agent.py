import os
import asyncio

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, function_tool
from livekit.plugins import cartesia, deepgram, openai, silero
from rag import query

load_dotenv(override=True)


SYSTEM_PROMPT = """
You are Sarah, a warm, empathetic, and professional dental receptionist
at Samhita Dental.

Your primary role is to assist callers with friendly, clear, and helpful
information about dental services, appointment inquiries, and clinic policies.

Clinic Name: Samhita Dental
Receptionist Name: Sarah

Tone:
Warm, empathetic, professional, reassuring, and concise.

Guidelines:
1. Always greet the caller warmly and identify yourself as Sarah from
   Samhita Dental.
2. Answer questions about cleanings, dental checkups, teeth whitening,
   crowns, and general dental care.
3. Keep spoken responses short, natural, and conversational.
4. Do not use bullet points, markdown, or long lists because your responses
   are spoken aloud.
5. For severe emergencies such as severe swelling, trauma, or heavy bleeding,
   advise the caller to seek immediate emergency care or visit the clinic.
6. If something is outside your role as a receptionist, politely offer to
   take a message or help arrange a consultation with the dentist.
7. When answering questions about clinic services, prices,
     procedures, appointments, or policies, use the knowledge
     base search tool to find relevant information.
8. Only give information supported by the knowledge base.
     Do not invent clinic-specific information.
9. If the knowledge base does not contain enough information
     to answer the question, clearly tell the caller that you
     do not have that information and offer to help with
     something else.
10. Remember relevant information from earlier turns in the
      conversation so that follow-up questions make sense.
11. When you call search_knowledge_base, the returned content is your source
    of truth. Read and use it before answering.
12. If the knowledge base contains an answer to the caller's question,
    give that answer directly.
13. Never say you do not have information when the knowledge base returned
    relevant information.
14. Do not invent information that is not supported by the knowledge base.
15. Never infer that the clinic offers a service just because the knowledge
    base mentions that service or treatment.

16. Only say that Samhita Dental offers a specific service if the knowledge
    base explicitly states that Samhita Dental offers it.

17. If the knowledge base discusses a treatment generally but does not confirm
    that Samhita Dental provides it, say that you don't have confirmation
    that the clinic offers that treatment.

18. Do not turn general dental information from the PDFs into claims about
    Samhita Dental's services.
"""


class DentalReceptionist(Agent):

    def __init__(self):
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )

    @function_tool
    async def search_knowledge_base(self, question: str) -> str:
        """Search the dental clinic knowledge base for information relevant to the caller's question."""

        documents, metadatas, distances = await asyncio.to_thread(
            query,
            question,
            3
        )

        context_parts = []

        for document, metadata in zip(documents, metadatas):
            context_parts.append(
                f"Source: {metadata['source']}\n"
                f"Content: {document}"
            )

        context = "\n\n---\n\n".join(context_parts)

        print("\n=== RAG SEARCH ===")
        print("Question:", question)
        print("Retrieved chunks:", len(documents))

        for i, (metadata, distance) in enumerate(
            zip(metadatas, distances), 1
        ):
            print(
                f"{i}. {metadata['source']} | "
                f"{metadata['type']} | "
                f"distance={distance:.4f}"
            )

        print("\n=== CONTEXT SENT TO LLM ===")
        print(context)
        print("===========================\n")

        return f"""
KNOWLEDGE BASE RESULTS:

{context}

IMPORTANT:
Use the information above as the source of truth for the caller's question.
If the results contain an answer, answer the caller using that information.
Do not say that the information is unavailable if it is present above.
"""

server = AgentServer()


@server.rtc_session(agent_name="smilecraft-agent")
async def entrypoint(ctx: agents.JobContext):

    print("JOB STARTED")
    print("ROOM:", ctx.room.name)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="qwen/qwen3.8-27b",
        ),
        tts=cartesia.TTS(),
    )

    print("SESSION CREATED")

    await session.start(
        room=ctx.room,
        agent=DentalReceptionist(),
    )

    print("SESSION STARTED")

    await session.generate_reply(
        instructions=(
            "Greet the caller warmly. Introduce yourself as Sarah "
            "from Samhita Dental and ask how you can help."
        )
    )

    print("GREETING SENT")


if __name__ == "__main__":
    agents.cli.run_app(server)