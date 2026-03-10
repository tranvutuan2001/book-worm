from langchain_core.callbacks.base import BaseCallbackHandler
import logging

# ANSI color codes for light blue
BLUE = "\033[94m"
RESET = "\033[0m"

class ColoredFormatter(logging.Formatter):
    """Formatter that colors only the message portion of the log in light blue."""
    def format(self, record: logging.LogRecord) -> str:
        # Preserve original message for safety
        original_msg = record.getMessage()
        # Temporarily replace the msg so %(message)s is colored by the base formatter
        record.msg = f"{BLUE}{original_msg}{RESET}"
        record.args = None
        try:
            return super().format(record)
        finally:
            # Restore original msg to avoid side effects if other handlers inspect the record
            record.msg = original_msg

# Get the LLM logger (configured by the main logging system)
logger = logging.getLogger('app.llm_connector')

class LLMLoggingHandler(BaseCallbackHandler):
    """Callback handler for logging all LLM and tool interactions"""

    # def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
    #     """Log when LLM starts"""
    #     logger.info("=" * 80)
    #     logger.info("🤖 LLM Starting")
    #     logger.info(f"Model: {serialized.get('name', 'unknown')}")
    #     logger.info(f"Number of prompts: {len(prompts)}")
    #     for idx, prompt in enumerate(prompts):
    #         logger.info(f"Prompt {idx + 1}: {prompt[:200]}...")
    #
    # def on_llm_end(self, response, **kwargs) -> None:
    #     """Log when LLM ends"""
    #     logger.info("✅ LLM Finished")
    #     if hasattr(response, 'generations'):
    #         for idx, gen in enumerate(response.generations):
    #             if gen:
    #                 text = gen[0].text if hasattr(gen[0], 'text') else str(gen[0])
    #                 logger.info(f"Generation {idx + 1}: {text[:200]}...")
    #
    # def on_llm_error(self, error: Exception, **kwargs) -> None:
    #     """Log when LLM errors"""
    #     logger.error(f"❌ LLM Error: {error}")

    def on_tool_start(self, serialized: dict[str, object], input_str: str, **kwargs: object) -> None:
        """Log when a tool starts"""
        logger.info("-" * 60)
        logger.info(f"🔧 Tool Called: {serialized.get('name', 'unknown')}")
        logger.info(f"Tool Input: {input_str[:500]}...")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log when a tool finishes"""
        logger.info(f"✓ Tool Output: {str(output)[:500]}...")
        logger.info("-" * 60)

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Log when a tool errors"""
        logger.error(f"❌ Tool Error: {error}")

    def on_agent_action(self, action, **kwargs) -> None:
        """Log agent actions"""
        logger.info(f"🎯 Agent Action: {action.tool}")
        logger.info(f"Action Input: {action.tool_input}")

    def on_agent_finish(self, finish, **kwargs) -> None:
        """Log when agent finishes"""
        logger.info(f"🏁 Agent Finished")
        logger.info(f"Final Output: {str(finish.return_values)[:300]}...")
        logger.info("=" * 80)

    def on_chain_start(self, serialized: dict[str, object], inputs: dict[str, object], **kwargs: object) -> None:
        """Log when a chain starts"""
        # Handle case where serialized might be None
        chain_name = "unknown"
        if serialized and isinstance(serialized, dict):
            chain_name = serialized.get('name', 'unknown')
        elif 'name' in kwargs:
            chain_name = kwargs['name']
        
        logger.info(f"⛓️  Chain Starting: {chain_name}")
        logger.info(f"Chain inputs: {str(inputs)[:200]}...")

    def on_chain_end(self, outputs: dict[str, object], **kwargs: object) -> None:
        """Log when a chain ends"""
        logger.info(f"⛓️  Chain Finished")
        logger.info(f"Chain outputs: {str(outputs)[:200]}...")

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Log when a chain errors"""
        logger.error(f"❌ Chain Error: {error}")
