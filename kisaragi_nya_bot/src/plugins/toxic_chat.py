import re
import os
import json
import time
import random
import asyncio
import requests
import chromadb
import base64
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from collections import defaultdict
from openai import AsyncOpenAI
from nonebot import on_message, get_driver
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot_plugin_apscheduler import scheduler
from nonebot.rule import to_me

MEMORY_FILE = "data/bot_memory.json"
os.makedirs("data", exist_ok=True)
driver = get_driver()
config = driver.config
client = AsyncOpenAI(
    base_url=getattr(config, "ollama_base_url", "http://localhost:11434/v1"),
    api_key=getattr(config, "ollama_api_key", "ollama"),
    timeout=60.0,
)
MODEL = getattr(config, "ollama_model", "qwen2.5:14b")
COOLDOWN_SECONDS = 5
MAX_HISTORY_ROUNDS = 3
ACTIVE_COOLDOWN = 60
SPEAK_PROBABILITY = 0.9
SENSITIVE_WORDS = []
GLOBAL_MSG_TIMESTAMPS = []
GLOBAL_MSG_LIMIT = 10
SLEEP_UNTIL_TIME = 0

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"user_profiles": {}, "affection": {}} # 格式: {"user_profiles": {"123456": "二次元"}}
global_memory = load_memory()
if "affection" not in global_memory:
    global_memory["affection"] = {}

chat_histories = defaultdict(list)
last_reply_time = defaultdict(float)
listen_buffer = defaultdict(list)
group_msg_counter = defaultdict(int)
last_active_time = defaultdict(float)
ANGRY_COUNT = defaultdict(int)

def save_memory(memory_data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent = 4)

def can_send_global():
    global GLOBAL_MSG_TIMESTAMPS
    current_time = time.time()
    GLOBAL_MSG_TIMESTAMPS = [t for t in GLOBAL_MSG_TIMESTAMPS if current_time - t < 60]

    if len(GLOBAL_MSG_TIMESTAMPS) >= GLOBAL_MSG_LIMIT:
        return False

    GLOBAL_MSG_TIMESTAMPS.append(current_time)
    return True

chroma_client = chromadb.PersistentClient(path = "data/chroma_db")

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "bge-m3"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            res = requests.post(self.url, json={"model": self.model_name, "prompt": text})
            res.raise_for_status()
            embeddings.append(res.json()["embedding"])
        return embeddings

fact_collection = chroma_client.get_or_create_collection(
    name = "group_facts",
    embedding_function = OllamaEmbeddingFunction() 
)

def retrieve_facts(user_id: str, query: str):
    """根据当前话题，去库里翻找这个人的黑历史"""
    try:
        results = fact_collection.query(
            query_text = [query],
            where = {"user_id": user_id},
            n_results = 2
        )
        if results["documents"] and results["documents"][0]:
            return "；".join(results["documents"][0])
    except Exception as e:
        pass
    return ""

async def extract_profile(user_id: str, user_msg: str):
    """悄悄分析用户的发言，提取标签并保存"""
    system_prompt = """你是一个心理学侧写师。请根据群友的完整发言，用一个极短的词语（不超过5个字）概括他的特征或身份，一定要全面，不可以武断。
只输出这个词语，绝对不要输出任何标点符号和解释。"""

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"该群友说：{user_msg}"}
            ],
            temperature=0.3,
            max_tokens=10
        )
        profile = response.choices[0].message.content.strip()
        profile = re.sub(r'[^\w\u4e00-\u9fa5]', '', profile)

        if profile:
            global_memory["user_profiles"][user_id] = profile
            print(f"【画像更新】用户 {user_id} -> {profile}")
    except Exception as e:
        print(f"画像提取失败: {e}")

async def extract_fact(user_id: str, user_msg: str):
    """判断这句话有没有事实，有的话存入向量库"""
    # 如果句子太短，大概率没实质内容，直接跳过节省算力
    if len(user_msg) < 6:
        return

    system_prompt = """你是一个事实提取器。请判断用户的这句话里，是否包含值得记住的客观事实（例如：他明天要考试、他买了新手机、他要去旅游、他失恋了（仅供参考，若是没有合适的客观事实也不要生搬硬套））。
如果有，请把事实浓缩成一句话（以“他”开头，例如：“他明天要考科目二”）。
如果没有实质事实（比如“哈哈哈”、“就这”），请只输出："无"。"""

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=30
        )
        fact = response.choices[0].message.content.strip()

        if fact and fact != "无":
            fact_collection.add(
                documents = [fact],
                metadatas = [{"user_id": user_id}],
                ids = [f"{user_id}_{int(time.time())}"]
            )
            print(f"【事实入库】{user_id}: {fact}")
    except Exception as e:
        print(f"事实提取失败: {e}")

async def analyze_affection(user_msg: str, reply: str):
    """后台分析：用户的这句话是夸我还是骂我？"""
    system_prompt = """你是一个情绪分析器。请判断【用户对AI说的话】是在：攻击/辱骂、正常聊天、还是讨好/夸奖？
只输出三个词之一：攻击、正常、讨好。不要输出任何标点。"""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户对你说：{user_msg}"}
            ],
            temperature=0.1,
            max_tokens=5
        )
        sentiment = response.choices[0].message.content.strip()
        return sentiment
    except :
        return "正常"

async def handle_image_toxic(event: GroupMessageEvent, user_msg: str, user_id: str):
    """处理带图片的 @ 消息"""
    group_id = event.group_id
    image_url = None
    for seg in event.get_message():
        if seg.type() == "image":
            image_url = seg.data.get("url")
            break

    if not image_url:
        return False

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        img_res = requests.get(image_url, headers = headers, timeout = 10)
        img_res.raise_for_status()
        img_base64 = base64.b64encode(img_res.content).decode('utf-8')

        user_profile = global_memory["user_profiles"].get(user_id, "未知路人")
        system_prompt = f"""你是一个名为“如月家的三小姐”的毒舌群友。
群友[{user_id}]（画像：{user_profile}）发了一张图片。
请你结合图片内容，用极度刻薄、阴阳怪气的网络烂梗嘲讽他。
要求：只输出一句不超过 20 个字的短句。以“喵”结尾。"""

        payload = {
            "model": "qwen2.5vl:7b", 
            "messages": [
                {
                    "role": "user",
                    "content": system_prompt + "\n请看图吐槽：",
                    "images": [img_base64]
                }
            ],
            "stream": False
        }

        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        )
        reply = response.json()["message"]["content"].strip()
        await group_listener.finish(reply)
        return True

    except Exception as e:
        await group_listener.finish(f"这图太瞎眼了，我懒得看喵（报错：{e}）")
        return True

async def handle_to_me(event: GroupMessageEvent, user_msg: str, user_id: str):
    global SLEEP_UNTIL_TIME
    group_id = event.group_id
    current_time = time.time()

    if current_time - last_reply_time[group_id] < COOLDOWN_SECONDS:
        return

    if not can_send_global():
        return 
    
    formatted_user_msg = f"群友[{user_id}]: {user_msg}"

    if listen_buffer[group_id]:
        chat_histories[group_id].append({"role": "user", "content": "刚才群里在聊：\n" + "\n".join(listen_buffer[group_id])})
        chat_histories[group_id].append({"role": "assistant", "content": "我都看在眼里，别管我。"})
        listen_buffer[group_id].clear()

    user_profile = global_memory["user_profiles"].get(user_id, "未知路人")
    affection = global_memory["affection"].get(user_id, 50) 
    tone = "极度刻薄、冷嘲热讽" if affection < 30 else ("稍微温和一点，但依然嘴硬" if affection > 70 else "普通的毒舌嘲讽")
    past_facts = retrieve_facts(user_id, user_msg)
    fact_prompt = f"\n【你突然想起他以前说过这些事】：{past_facts}\n请结合这些黑历史精准吐槽！" if past_facts else ""

    system_prompt = """你是一个名为“如月家的三小姐”的毒舌群友。你高冷、刻薄、神经质、又容易破防，喜欢用网络烂梗嘲讽别人，自己有一点吃亏了就要大喊“呜呜呜”。你正在回复群友[{user_id}]。你偷偷看了一眼他的画像，发现他是个：【{user_profile}】。{fact_prompt} 你对他的好感度是：{affection}/100。请保持【{tone}】的语气。

    【重要指令】：
如果你觉得一句话不够，请用 "||" 符号分隔你的回复，分成 2 到 3 条短句发送（模拟群聊连发）。
不要输出任何完整的复述，不要输出解释，严格按照这个格式！
例如（仅参考格式，不要参考句子内容）：就这？||我还以为多狠呢||典中典
每条短句绝对不要超过 15 个字。
    
    要求：
0. 被问及名字或被问到你是谁，说：“你爹都不认识了？”
1. 嘲讽一句。
2. 回复必须极短，不超过30个字，要像真实在群里水群的人，结合之前的聊天记录来吐槽。
3. 不要说“作为AI”、“很高兴为您服务”这种废话。
4. 看见亲西方言论，说：“别舔你白爹了”
5. 有人夸奖你或者表扬你，要开心，得意洋洋，用可爱的语气回复，保持傲娇。 
6. 对于剩下的没有列举到的情况，如果对方说的话很蠢，或者对方说的话具有攻击性，令人不满，说“野狗”。
7. 经常使用“典”、“绷不住了”、“就这？”、“差不多得了”等网络用语，但不要每句话都用，而且这一条优先级低于4、5、6、7条。
8. 句子结尾都带个“喵”或者“w”
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_histories[group_id])
    messages.append({"role": "user", "content": formatted_user_msg})
    
    reply = ""
    try:
        # 发送请求给本地模型
        response = await client.chat.completions.create(
            model = MODEL, messages = messages, temperature = 0.8, max_tokens = 50
        )
        reply = response.choices[0].message.content.strip()
        reply = re.sub(r'^["\'「]|["\'」]$', '', reply)

        chat_histories[group_id].append({"role": "user", "content": formatted_user_msg})
        chat_histories[group_id].append({"role": "assistant", "content": reply})

        if len(chat_histories[group_id]) > MAX_HISTORY_ROUNDS * 2:
            chat_histories[group_id] = chat_histories[group_id][-MAX_HISTORY_ROUNDS * 2:]

        last_reply_time[group_id] = current_time
        sentiment = await analyze_affection(user_msg, reply)
        if sentiment == "攻击":
            affection = max(0, affection - 10) # 好感度扣 10 分，最低 0
            ANGRY_COUNT[user_id] += 1

            if ANGRY_COUNT[user_id] >= 3:
                reply = "行，你们行。咱不伺候了，自闭半小时。"
                SLEEP_UNTIL_TIME = time.time() + 1800 # 1800秒 = 30分钟
                ANGRY_COUNT[user_id] = 0 # 怒气值清零
        elif sentiment == "讨好":
            affection = min(100, affection + 5) # 好感度加 5 分，最高 100
            ANGRY_COUNT[user_id] = 0
        global_memory["affection"][user_id] = affection
        
    except Exception as e:
        reply = f"懒得理你，打胶去了喵（报错：{str(e)}）"

    parts = [p.strip() for p in re.split(r'\|\||｜｜|\||｜', reply) if p.strip()]

    if not parts:
        parts = [reply]

    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            await group_listener.send(part)
            await asyncio.sleep(random.uniform(0.5, 1.5))
        else:
            await group_listener.finish(part) # 这里已经改对了

group_listener = on_message(priority = 10, block = False)

@group_listener.handle()
async def _(event: GroupMessageEvent):
    global SLEEP_UNTIL_TIME
    group_id = event.group_id
    user_id = event.user_id
    user_msg = event.get_plaintext().strip()
    # if not user_msg:
        # await toxic_chat.finish("干嘛？光@我不说话？想艾草吗？")

    if not user_msg or user_id == event.self_id:
        return

    if time.time() < SLEEP_UNTIL_TIME:
        return 

    if any(word in user_msg for word in SENSITIVE_WORDS):
        print(f"[风控] 检测到敏感词，已自动装死: {user_msg}")
        return

    listen_buffer[group_id].append(f"群友[{user_id}]: {user_msg}")
    if len(listen_buffer[group_id]) > 5:
        listen_buffer[group_id].pop(0)

    if not event.is_tome():
        asyncio.create_task(extract_profile(user_id, user_msg))
        asyncio.create_task(extract_fact(user_id, user_msg))

    if event.is_tome():
        if any(seg.type in ["image", "mface"] for seg in event.get_message()):
            await handle_image_toxic(event, user_msg, user_id)
            return 
        await handle_to_me(event, user_msg, user_id)
        return

    group_msg_counter[group_id] += 1
    current_time = time.time()

    if (current_time - last_active_time[group_id] > ACTIVE_COOLDOWN) and group_msg_counter[group_id] >= 3:
        if random.random() < SPEAK_PROBABILITY:
            # 触发主动水群！
            group_msg_counter[group_id] = 0 # 计数器清零
            last_active_time[group_id] = current_time # 更新冷却时间
            context_text = "\n".join(listen_buffer[group_id])
            system_prompt = """你是一个名为“如月家的三小姐”的毒舌群友。你高冷、刻薄、神经质、又容易破防，喜欢用网络烂梗嘲讽别人，自己有一点吃亏了就要大喊“呜呜呜”。你刚才一直在潜水看他们聊天。
现在你实在忍不住了，决定插一句话吐槽他们。
要求：
1. 结合他们刚才聊天的内容（见下文）进行毒舌点评。
2. 极短，不超过 20 个字。
3. 高冷、刻薄、搞笑、幽默。
4. 句子结尾都带个“喵”或者“w”
"""

            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"大家刚才在聊：\n{context_text}\n\n请你插话吐槽。"}
                    ],
                    temperature=0.9,
                    max_tokens=40
                )
                reply = response.choices[0].message.content.strip()
                reply = re.sub(r'^["\'「]|["\'」]$', '', reply)
                
                # 主动发言不需要 finish，用 send
                await group_listener.send(reply)
            except Exception:
                pass # 主动插话失败就算了，不报错

@scheduler.scheduled_job("interval", seconds = 60)
async def auto_save_memory():
    try:
        save_memory(global_memory)
        print("[系统] 记忆库已自动落盘")
    except Exception as e:
        print(f"[错误] 自动保存记忆库失败: {e}")

@driver.on_shutdown
async def on_shutdown():
    save_memory(global_memory)
    print("[系统] 机器人即将关闭，记忆库已成功保存至硬盘！")
