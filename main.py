import discord
import os
import logging
import asyncio
import aiohttp
from discord.ext import commands
from dotenv import load_dotenv
from typing import Optional
import json

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración del bot
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/DeepSeek-V3')
SPECIFIC_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')

# Configuración de intents de Discord
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

# Inicializar el cliente de Discord
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# URL de la API de Hugging Face para el modelo DeepSeek-V3 (usando el endpoint de inferencia)
HF_API_URL = f"https://api-inference.huggingface.co/models/{DEEPSEEK_MODEL}"
HF_HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
    "Content-Type": "application/json"
}

# Almacenar historial de conversaciones
conversation_history = {}
MAX_HISTORY = 10

# Semáforo para controlar procesamiento concurrente
bot.is_processing = False

def get_channel_id() -> Optional[int]:
    """Obtener el ID del canal específico si está configurado"""
    if SPECIFIC_CHANNEL_ID and SPECIFIC_CHANNEL_ID.isdigit():
        return int(SPECIFIC_CHANNEL_ID)
    return None

async def query_deepseek(prompt: str, user_id: Optional[int] = None) -> str:
    """
    Consultar DeepSeek-V3 usando la API de Hugging Face directamente
    """
    global conversation_history
    
    try:
        # Preparar el payload para la API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        logger.info(f"Consultando a DeepSeek: {prompt[:50]}...")
        
        # Usar aiohttp para la petición asíncrona
        async with aiohttp.ClientSession() as session:
            async with session.post(HF_API_URL, headers=HF_HEADERS, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Procesar la respuesta según el formato
                    if isinstance(result, list) and len(result) > 0:
                        if 'generated_text' in result[0]:
                            ai_response = result[0]['generated_text']
                        else:
                            ai_response = str(result[0])
                    elif isinstance(result, dict) and 'generated_text' in result:
                        ai_response = result['generated_text']
                    else:
                        ai_response = str(result)
                    
                    logger.info(f"Respuesta recibida: {ai_response[:50]}...")
                    
                    # Actualizar historial si hay user_id
                    if user_id:
                        if user_id not in conversation_history:
                            conversation_history[user_id] = []
                        
                        conversation_history[user_id].append({
                            "user": prompt,
                            "assistant": ai_response
                        })
                        
                        # Mantener solo últimos MAX_HISTORY mensajes
                        if len(conversation_history[user_id]) > MAX_HISTORY:
                            conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]
                    
                    return ai_response
                    
                elif response.status == 503:
                    # El modelo está cargándose
                    logger.info("Modelo cargándose, esperando...")
                    await asyncio.sleep(5)
                    return await query_deepseek(prompt, user_id)  # Reintentar
                else:
                    error_text = await response.text()
                    logger.error(f"Error HTTP {response.status}: {error_text}")
                    
                    # Mensajes de error más amigables
                    if "loading" in error_text.lower():
                        return "🔄 El modelo se está cargando. Por favor, espera unos segundos y vuelve a intentarlo."
                    elif "rate limit" in error_text.lower():
                        return "⏳ Hay muchas solicitudes en este momento. Por favor, espera un momento y vuelve a intentar."
                    else:
                        return f"❌ Error al procesar tu mensaje. Código: {response.status}"
                    
    except asyncio.TimeoutError:
        logger.error("Timeout en la petición")
        return "⌛ La petición tardó demasiado. Por favor, intenta de nuevo."
    except Exception as e:
        logger.error(f"Error al consultar DeepSeek: {e}", exc_info=True)
        return f"❌ Lo siento, ocurrió un error: {str(e)[:100]}"

# Alternativa: Usar el endpoint de chat completions (más moderno)
async def query_deepseek_chat(prompt: str, user_id: Optional[int] = None) -> str:
    """
    Versión alternativa usando el formato de chat completions
    """
    try:
        # Construir el historial de conversación
        messages = []
        
        # Mensaje del sistema
        messages.append({
            "role": "system",
            "content": "Eres un asistente útil, amigable y servicial. Responde en el mismo idioma en que te hablan."
        })
        
        # Agregar historial si existe
        if user_id and user_id in conversation_history:
            for exchange in conversation_history[user_id][-3:]:  # Últimos 3 intercambios
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Agregar mensaje actual
        messages.append({"role": "user", "content": prompt})
        
        # Payload para chat completions
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(HF_API_URL, headers=HF_HEADERS, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Extraer respuesta
                    if isinstance(result, list):
                        ai_response = result[0].get('generated_text', str(result[0]))
                    else:
                        ai_response = result.get('generated_text', str(result))
                    
                    # Actualizar historial
                    if user_id:
                        if user_id not in conversation_history:
                            conversation_history[user_id] = []
                        
                        conversation_history[user_id].append({
                            "user": prompt,
                            "assistant": ai_response
                        })
                        
                        if len(conversation_history[user_id]) > MAX_HISTORY:
                            conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]
                    
                    return ai_response
                else:
                    error_text = await response.text()
                    logger.error(f"Error {response.status}: {error_text}")
                    return f"❌ Error {response.status}. Por favor, intenta de nuevo."
                    
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        return f"❌ Error: {str(e)[:100]}"

# Usar esta función por defecto (puedes cambiar entre query_deepseek y query_deepseek_chat)
query_function = query_deepseek  # Cambia a query_deepseek_chat si prefieres

@bot.event
async def on_ready():
    """Evento cuando el bot está listo"""
    logger.info(f'✅ Bot conectado como {bot.user.name}')
    logger.info(f'🆔 ID del bot: {bot.user.id}')
    logger.info(f'📝 Usando modelo: {DEEPSEEK_MODEL}')
    
    # Verificar conexión con Hugging Face
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://huggingface.co/api/models/{DEEPSEEK_MODEL}") as response:
                if response.status == 200:
                    logger.info("✅ Conexión con Hugging Face exitosa")
                else:
                    logger.warning("⚠️ No se pudo verificar el modelo en Hugging Face")
    except Exception as e:
        logger.warning(f"⚠️ Error al verificar Hugging Face: {e}")
    
    # Establecer estado
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="!help | DeepSeek V3"
        )
    )

@bot.event
async def on_message(message):
    """Manejador de mensajes"""
    
    if message.author == bot.user:
        return
    
    # Verificar canal específico
    channel_id = get_channel_id()
    if channel_id and message.channel.id != channel_id:
        await bot.process_commands(message)
        return
    
    # Evitar procesamiento múltiple
    if bot.is_processing:
        return
    
    # Verificar si debemos responder
    is_bot_mentioned = bot.user.mentioned_in(message)
    is_direct_message = isinstance(message.channel, discord.DMChannel)
    is_specific_channel = channel_id and message.channel.id == channel_id
    
    # Comandos con prefijo !
    if message.content.startswith('!'):
        await bot.process_commands(message)
        return
    
    # Responder en canal específico
    if is_specific_channel:
        bot.is_processing = True
        try:
            async with message.channel.typing():
                response = await query_function(message.content, message.author.id)
                
                # Manejar respuestas largas
                if len(response) <= 2000:
                    await message.reply(response)
                else:
                    chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                    await message.reply(f"📝 **Respuesta larga** (1/{len(chunks)}):")
                    for i, chunk in enumerate(chunks, 1):
                        await message.channel.send(chunk)
                        if i < len(chunks):
                            await asyncio.sleep(1)
        finally:
            bot.is_processing = False
    
    # Responder a menciones y MD
    elif is_bot_mentioned or is_direct_message:
        bot.is_processing = True
        try:
            # Limpiar mensaje de menciones
            clean_content = message.content
            if is_bot_mentioned:
                for mention in message.mentions:
                    clean_content = clean_content.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
                clean_content = clean_content.strip()
            
            if not clean_content:
                await message.reply("¿En qué puedo ayudarte?")
                return
            
            async with message.channel.typing():
                response = await query_function(clean_content, message.author.id)
                
                if len(response) <= 2000:
                    await message.reply(response)
                else:
                    chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                    await message.reply(f"📝 **Respuesta larga** (1/{len(chunks)}):")
                    for i, chunk in enumerate(chunks, 1):
                        await message.channel.send(chunk)
                        if i < len(chunks):
                            await asyncio.sleep(1)
        finally:
            bot.is_processing = False
    
    else:
        await bot.process_commands(message)

@bot.command(name='help')
async def help_command(ctx):
    """Comando de ayuda"""
    embed = discord.Embed(
        title="🤖 Asistente DeepSeek V3",
        description="Bot de Discord con IA usando DeepSeek-V3 de Hugging Face",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="📝 **Comandos**",
        value=(
            "`!help` - Muestra esta ayuda\n"
            "`!clear` - Limpia tu historial\n"
            "`!modelo` - Información del modelo\n"
            "`!estado` - Estado del bot\n"
            "`!probar` - Prueba simple"
        ),
        inline=False
    )
    
    embed.add_field(
        name="💬 **Cómo usar**",
        value=(
            "• **Mencióname**: @bot ¿pregunta?\n"
            "• **Mensaje directo**: Envíame un MD\n"
            f"{'• **Canal específico**: Escribe directamente' if channel_id else ''}"
        ),
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='clear')
async def clear_history(ctx):
    """Limpiar historial"""
    global conversation_history
    if ctx.author.id in conversation_history:
        del conversation_history[ctx.author.id]
        await ctx.reply("✅ Historial limpiado")
    else:
        await ctx.reply("📭 No hay historial")

@bot.command(name='modelo')
async def show_model(ctx):
    """Mostrar información del modelo"""
    embed = discord.Embed(
        title="🤖 Modelo Actual",
        description=f"**{DEEPSEEK_MODEL}**",
        color=discord.Color.green()
    )
    embed.add_field(name="Tipo", value="Conversational AI")
    embed.add_field(name="API", value="Hugging Face Inference")
    await ctx.send(embed=embed)

@bot.command(name='estado')
async def check_status(ctx):
    """Verificar estado del bot"""
    embed = discord.Embed(
        title="📊 Estado del Bot",
        color=discord.Color.gold()
    )
    
    embed.add_field(name="Discord", value="✅ Conectado", inline=True)
    embed.add_field(name="Latencia", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Usuarios activos", value=f"{len(conversation_history)}", inline=True)
    
    # Probar Hugging Face
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://huggingface.co/api/models/{DEEPSEEK_MODEL}") as response:
                if response.status == 200:
                    embed.add_field(name="Hugging Face", value="✅ Conectado", inline=True)
                else:
                    embed.add_field(name="Hugging Face", value="⚠️ Error", inline=True)
    except:
        embed.add_field(name="Hugging Face", value="❌ Desconectado", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='probar')
async def test_command(ctx):
    """Comando de prueba simple"""
    await ctx.reply("✅ El bot está funcionando correctamente!")

@bot.event
async def on_command_error(ctx, error):
    """Manejador de errores"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.reply("❌ Comando no encontrado. Usa `!help`")
    else:
        logger.error(f"Error: {error}")
        await ctx.reply(f"❌ Error: {str(error)[:100]}")

# Iniciar el bot
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("❌ No se encontró DISCORD_TOKEN")
        exit(1)
    
    if not HUGGINGFACE_TOKEN:
        logger.error("❌ No se encontró HUGGINGFACE_TOKEN")
        exit(1)
    
    logger.info("🚀 Iniciando bot...")
    bot.run(DISCORD_TOKEN)