"""
使用配置文件运行数据生成的示例脚本
"""

import asyncio
import json
from pathlib import Path
from generate_training_data import DataAugmentationPipeline


async def main():
    # 加载配置文件
    config_file = Path("config.json")

    if not config_file.exists():
        print("Error: config.json not found")
        print("Please copy config.example.json to config.json and update the settings")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 提取配置
    api_settings = config.get("api_settings", {})
    data_settings = config.get("data_settings", {})

    # 创建管道
    pipeline = DataAugmentationPipeline(
        api_key=api_settings.get("api_key"),
        base_url=api_settings.get("base_url"),
        model=api_settings.get("model", "gpt-4o-mini"),
        max_concurrent=api_settings.get("max_concurrent", 10),
        output_dir=data_settings.get("output_dir", "./training_data")
    )

    # 处理数据集
    await pipeline.process_all_datasets(
        data_dir=data_settings.get("data_dir", "../../processed_data"),
        datasets=data_settings.get("datasets")
    )


if __name__ == "__main__":
    asyncio.run(main())
