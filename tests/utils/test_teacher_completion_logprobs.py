"""Tests for token-in-token-out teacher logprob alignment."""

import unittest

import torch

from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.gkd_helpers import (align_teacher_routes_to_completion_turns,
                                             assemble_teacher_completion_logprobs)
from swift.rlhf_trainers.utils import (build_completion_turn_token_ids, build_response_token_mask,
                                       replace_assistant_response_with_ids)


def parsed(token_ids):
    return ([[float(-token_id)] for token_id in token_ids], [[token_id] for token_id in token_ids])


class TestTeacherCompletionLogprobs(unittest.TestCase):

    def test_multi_turn_spans_with_tool_tokens_and_loss_mask(self):
        # Two raw assistant spans are separated by a tool response. The first token of
        # turn 2 is a non-thinking prefix and must not contribute to the completion loss.
        teacher_ids = [10, 11, 101, 102, 20, 21, 201, 202, 203, 30]
        completion_mask = torch.tensor([[0, 1, 0, 1, 1, 0, 1]], dtype=torch.bool)
        output = assemble_teacher_completion_logprobs(
            [parsed(teacher_ids)],
            completion_mask,
            torch.device('cpu'),
            response_token_ids=[[[101, 102], [201, 202, 203]]],
            response_loss_mask=[[[1, 1], [0, 1, 1]]],
        )

        active = completion_mask[0].nonzero(as_tuple=True)[0]
        self.assertEqual(output.topk_indices[0, active, 0].tolist(), [101, 102, 202, 203])
        self.assertEqual(output.topk_logprobs[0, active, 0].tolist(), [-101.0, -102.0, -202.0, -203.0])

    def test_includes_template_boundaries_after_each_assistant_turn(self):
        # Qwen-like two-token assistant boundaries are active labels; tool
        # observations between assistant turns remain masked.
        teacher_ids = [10, 101, 102, 900, 901, 20, 21, 201, 202, 900, 901, 30]
        completion_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        output = assemble_teacher_completion_logprobs(
            [parsed(teacher_ids)],
            completion_mask,
            torch.device('cpu'),
            response_token_ids=[[[101, 102], [201, 202]]],
            response_loss_mask=[[[1, 1], [1, 1]]],
            completion_turn_token_ids=[[[101, 102, 900, 901], [201, 202, 900, 901]]],
        )
        self.assertEqual(output.topk_indices[0, :, 0].tolist(), [101, 102, 900, 901, 201, 202, 900, 901])

    def test_teacher_routes_include_boundaries_without_shifting_later_turns(self):
        routes = align_teacher_routes_to_completion_turns(
            route_turns=[['image', 'image']],
            response_loss_mask=[[1, 1], [1, 1]],
            completion_turn_token_ids=[[101, 102, 900, 901], [201, 202, 900, 901]],
            default_route='video')
        self.assertEqual(routes, ['image'] * 4 + ['video'] * 4)

    def test_student_labels_define_turn_boundaries(self):
        sample = OnPolicySample(
            messages=[],
            response_token_ids=[[101, 102], [201, 202]],
            response_loss_mask=[[1, 1], [1, 1]],
            encoded={
                'input_ids': [10, 101, 102, 900, 901, 20, 201, 202, 900, 901],
                'labels': [-100, 101, 102, 900, 901, -100, 201, 202, 900, 901],
            })
        completion_mask = torch.ones(1, 8, dtype=torch.bool)
        self.assertEqual(
            build_completion_turn_token_ids([sample], completion_mask),
            [[[101, 102, 900, 901], [201, 202, 900, 901]]])

    def test_branch_mask_defaults_missing_final_turn_to_zero(self):
        completion_mask = torch.ones(1, 8, dtype=torch.bool)
        result = build_response_token_mask(
            nested_masks=[[[1, 0]]],
            completion_mask=completion_mask,
            device=torch.device('cpu'),
            response_loss_masks=[[[1, 1], [1, 1]]],
            completion_turn_token_ids=[[[101, 102, 900, 901], [201, 202, 900, 901]]])
        self.assertEqual(result[0].tolist(), [True, False, False, False, False, False, False, False])

    def test_prefix_injection_does_not_mutate_raw_rollout_tokens(self):
        token_ids = [[101, 102]]
        loss_mask = [[1, 1]]
        messages = [{'role': 'assistant', 'content': 'answer'}]
        replaced = replace_assistant_response_with_ids(
            messages, token_ids, loss_mask, non_thinking_prefix_ids=[700, 701])
        self.assertEqual(token_ids, [[101, 102]])
        self.assertEqual(loss_mask, [[1, 1]])
        self.assertEqual(replaced[0]['content']['token_ids'], [700, 701, 101, 102])
        self.assertEqual(replaced[0]['content']['loss_scale'], [0, 0, 1, 1])

    def test_matches_latest_occurrence_not_same_tokens_in_prompt(self):
        teacher_ids = [101, 102, 9, 9, 101, 102, 30]
        completion_mask = torch.tensor([[1, 1]], dtype=torch.bool)
        output = assemble_teacher_completion_logprobs(
            [parsed(teacher_ids)],
            completion_mask,
            torch.device('cpu'),
            response_token_ids=[[[101, 102]]],
            response_loss_mask=[[[1, 1]]],
        )
        self.assertEqual(output.topk_indices[0, :, 0].tolist(), [101, 102])

    def test_three_turn_spans_with_two_tool_responses(self):
        teacher_ids = [10, 101, 102, 20, 21, 201, 202, 30, 31, 301, 302, 303, 40]
        completion_mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1]], dtype=torch.bool)
        output = assemble_teacher_completion_logprobs(
            [parsed(teacher_ids)],
            completion_mask,
            torch.device('cpu'),
            response_token_ids=[[[101, 102], [201, 202], [301, 302, 303]]],
            response_loss_mask=[[[1, 0], [1, 1], [0, 1, 1]]],
        )

        active = completion_mask[0].nonzero(as_tuple=True)[0]
        self.assertEqual(output.topk_indices[0, active, 0].tolist(), [101, 201, 202, 302, 303])
        self.assertEqual(output.topk_logprobs[0, active, 0].tolist(), [-101.0, -201.0, -202.0, -302.0, -303.0])

    def test_rejects_turn_mask_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            assemble_teacher_completion_logprobs(
                [parsed([10, 101, 102, 30])],
                torch.tensor([[1, 1]], dtype=torch.bool),
                torch.device('cpu'),
                response_token_ids=[[[101, 102]]],
                response_loss_mask=[[[1]]],
            )


if __name__ == '__main__':
    unittest.main()
