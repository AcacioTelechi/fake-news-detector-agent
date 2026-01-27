from typing import Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import TavilyClient


class ModelConfig(BaseModel):
    """
    Configuração para um modelo específico de um node.
    """
    model: str = Field(description="Nome do modelo (ex: gpt-3.5-turbo, gpt-4, ollama:llama2)")
    temperature: float = Field(default=0.0, description="Temperatura para geração")
    model_provider: Optional[str] = Field(default=None, description="Provider do modelo (ex: openai, ollama)")
    base_url: Optional[str] = Field(default=None, description="URL base para o modelo (ex: http://localhost:11434)")


# Tipo que aceita string (nome do modelo) ou configuração completa
ModelConfigType = Union[str, ModelConfig]


def _normalize_config(value: ModelConfigType, default_temperature: float = 0.0) -> ModelConfig:
    """Normaliza string ou ModelConfig para ModelConfig."""
    if isinstance(value, str):
        return ModelConfig(model=value, temperature=default_temperature)
    return value


class ModelsRegistry(BaseModel):
    """
    Registry que armazena configurações de modelos para cada node e instancia BaseChatModel sob demanda.
    
    Cada node pode receber:
    - Uma string com o nome do modelo (usa temperatura global)
    - Um ModelConfig com configurações específicas
    
    Exemplo:
        ModelsRegistry(
            entry="gpt-3.5-turbo",  # usa temperatura global
            planner=ModelConfig(model="gpt-4", temperature=0.7),  # config específica
            researcher="gpt-3.5-turbo",
            analyst=ModelConfig(model="gpt-4-turbo", temperature=0.0),
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configurações de modelo para cada node
    entry: ModelConfigType = Field(default="gpt-3.5-turbo")
    planner: ModelConfigType = Field(default="gpt-3.5-turbo")
    researcher: ModelConfigType = Field(default="gpt-3.5-turbo")
    analyst: ModelConfigType = Field(default="gpt-3.5-turbo")

    # Temperatura global (usada quando node recebe apenas string)
    default_temperature: float = Field(default=0.0)

    # Cache de modelos instanciados
    _models_cache: dict[str, BaseChatModel] = {}

    def _get_node_config(self, node_name: str) -> ModelConfig:
        """Retorna a configuração normalizada para um node."""
        config = getattr(self, node_name, None)
        if config is None:
            raise ValueError(f"Node '{node_name}' not found in models registry")
        return _normalize_config(config, self.default_temperature)

    def _create_model(self, config: ModelConfig) -> BaseChatModel:
        """Cria uma instância de BaseChatModel a partir da configuração."""
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.model_provider:
            kwargs["model_provider"] = config.model_provider
        if config.base_url:
            kwargs["base_url"] = config.base_url

        return init_chat_model(**kwargs)

    def _get_cache_key(self, config: ModelConfig) -> str:
        """Gera chave de cache única para uma configuração."""
        return f"{config.model}:{config.temperature}:{config.model_provider}:{config.base_url}"

    def get_model(self, node_name: str) -> BaseChatModel:
        """
        Retorna o BaseChatModel para um node específico.
        Instancia o modelo sob demanda e armazena em cache.
        
        Args:
            node_name: Nome do node (entry, planner, researcher, analyst)
            
        Returns:
            BaseChatModel instanciado para o node
        """
        config = self._get_node_config(node_name)
        cache_key = self._get_cache_key(config)
        
        if cache_key not in self._models_cache:
            self._models_cache[cache_key] = self._create_model(config)

        return self._models_cache[cache_key]

    def get_model_name(self, node_name: str) -> str:
        """Retorna o nome do modelo configurado para um node."""
        config = self._get_node_config(node_name)
        return config.model

    def get_model_config(self, node_name: str) -> ModelConfig:
        """Retorna a configuração completa do modelo para um node."""
        return self._get_node_config(node_name)


class RuntimeContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    models_registry: ModelsRegistry = Field(default_factory=ModelsRegistry)
    tavily: TavilyClient = Field(default=None)
