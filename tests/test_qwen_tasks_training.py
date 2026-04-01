import pytest

torch = pytest.importorskip("torch")

from kabr_vlm.training.qwen_tasks import (
    QwenTaskDataset,
    QwenTaskCollator,
    QwenTaskExample,
    build_task_messages,
    task_prompt,
)


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = messages[0]["content"][-1]["text"]
        image_count = sum(1 for item in messages[0]["content"] if item["type"] == "image")
        answer = ""
        if len(messages) > 1:
            answer = messages[1]["content"][0]["text"]
        suffix = "<gen>" if add_generation_prompt else ""
        return f"img={image_count}|prompt={prompt}|answer={answer}{suffix}"

    def __call__(self, text, images, return_tensors, padding):
        rows = [[len(token) for token in item.split("|")] for item in text]
        max_len = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            padded = row + [0] * (max_len - len(row))
            mask = [1] * len(row) + [0] * (max_len - len(row))
            input_ids.append(padded)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": torch.ones(len(images), 4, dtype=torch.float32),
            "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
        }


def test_task_prompt_supports_behavior() -> None:
    prompt = task_prompt("action_recognition")
    assert "behavior label" in prompt
    assert "Walk" in prompt
    assert "Occluded" in prompt


def test_qwen_task_dataset_parses_structured_behavior_answer() -> None:
    dataset = QwenTaskDataset(
        [
            {
                "id": "clip-1",
                "image_paths": ["/home/jhyan/KABR_files/dataset/image/G0145.2/66.jpg"],
                "task": "action_recognition",
                "prompt": 'Reply with code: recognize(entity_type="action", value="<label>").',
                "answer": 'recognize(entity_type="action", value="Walk")',
                "metadata": {},
            }
        ]
    )

    assert dataset[0].label == "Walk"


def test_build_task_messages_keeps_ordered_images() -> None:
    image_paths = ["a.jpg", "b.jpg", "c.jpg"]
    messages = build_task_messages(task_prompt("animal_recognition"), image_paths, answer="giraffe")

    assert messages[0]["role"] == "user"
    assert [item["image"] for item in messages[0]["content"] if item["type"] == "image"] == image_paths
    assert messages[1]["content"][0]["text"] == "giraffe"


def test_qwen_task_collator_masks_prompt_tokens() -> None:
    collator = QwenTaskCollator(
        processor=FakeProcessor(),
        max_seq_length=128,
    )
    batch = collator(
        [
            QwenTaskExample(
                example_id="clip-1",
                image_paths=["/home/jhyan/KABR_files/dataset/image/G0145.2/66.jpg"],
                prompt=task_prompt("animal_recognition"),
                answer="giraffe",
                label="giraffe",
                task_mode="animal_recognition",
                metadata={},
            ),
            QwenTaskExample(
                example_id="clip-2",
                image_paths=["/home/jhyan/KABR_files/dataset/image/G0145.2/67.jpg"],
                prompt=task_prompt("animal_recognition"),
                answer="giraffe",
                label="giraffe",
                task_mode="animal_recognition",
                metadata={},
            ),
        ]
    )

    assert "labels" in batch
    assert batch["labels"].shape == batch["input_ids"].shape
    assert torch.any(batch["labels"] == -100)
    assert batch["pixel_values"].shape[0] == 2


def test_qwen_task_collator_supports_behavior() -> None:
    collator = QwenTaskCollator(
        processor=FakeProcessor(),
        max_seq_length=128,
    )
    batch = collator(
        [
            QwenTaskExample(
                example_id="clip-1",
                image_paths=["/home/jhyan/KABR_files/dataset/image/G0145.2/66.jpg"],
                prompt=task_prompt("action_recognition"),
                answer="Walk",
                label="Walk",
                task_mode="action_recognition",
                metadata={},
            )
        ]
    )

    assert batch["labels"].shape[0] == 1


def test_qwen_task_collator_supports_structured_answer() -> None:
    collator = QwenTaskCollator(
        processor=FakeProcessor(),
        max_seq_length=128,
    )
    batch = collator(
        [
            QwenTaskExample(
                example_id="clip-1",
                image_paths=["/home/jhyan/KABR_files/dataset/image/G0145.2/66.jpg"],
                prompt='Reply with code: recognize(entity_type="action", value="<label>").',
                answer='recognize(entity_type="action", value="Walk")',
                label="Walk",
                task_mode="action_recognition",
                metadata={},
            )
        ]
    )

    assert batch["labels"].shape[0] == 1
