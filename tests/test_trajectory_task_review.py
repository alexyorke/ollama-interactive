from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

from scripts import trajectory_task_review as review


class TrajectoryTaskReviewPureTests(unittest.TestCase):
    def test_build_parser_accepts_legacy_output_flag(self) -> None:
        args = review._build_parser().parse_args(["--output", "scratch/external/datasets/task-review.json"])

        self.assertEqual(args.output, Path("scratch/external/datasets/task-review.json"))

    def test_resolve_output_paths_from_legacy_output_json_path(self) -> None:
        output_json, output_md = review._resolve_output_paths(
            output=Path("scratch/external/datasets/task-review.json"),
            output_json=None,
            output_md=None,
        )

        self.assertEqual(output_json, Path("scratch/external/datasets/task-review.json"))
        self.assertEqual(output_md, Path("scratch/external/datasets/task-review.md"))

    def test_classify_task_family_prefers_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Fix the failing parser and rerun the tests.",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_lora_application_check_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you check this task to see if it properly applid the lora: efc992cb-cb85-4a6f-9ce8-1341320c2e4b",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_sense_check_scan_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you run the deslopify scan and then sense-check the results",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_why_did_this_fail_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Why did this fail? fa75b4f0-a6e5-484f-9dc6-118ed05a7ba9",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_why_task_failed_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you try to undersatnd why this task failed? See logs: 0ebf307a-e422-4ac1-807e-07f0f1118c8e",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_spring_console_log_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "depuis un moment au run de mon app spring je n ai plus les logs console "
                "de maniere colore et organise comme par defaut"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_eureka_status_dump_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Instances currently registered with Eureka Application AMIs Availability Zones Status "
                "AWDPAY-V2-API n/a (1) (1) UP (1) - awdpay-v2-api:8081 SERVICE-GATEWAY n/a (1) (1) "
                "UP (1) - 192.168.100.8:service-gateway:8080 General Info"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_logback_spring_console_style_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "les logs sur ma console n ont plus de style pas de couleur depuis le fichier "
                "@src/main/resources/logback-spring.xml"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_queued_reset_diagnosis_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you see why this task keeps getting reset back to Queued: 66a8c661-4cd4-45be-a7bc-fdc9559caba3",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_doesnt_seem_to_work_ui_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see in the gallery on iPad tapping an image or even double-tapping it to open it "
                "doesn't seem to work for some reason. Maybe the click handlers get blocked by something else. "
                "I'm not sure."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_prefers_feature_over_setup_build_for_concrete_product_request(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Build a completely frontend-only webapp that takes input UPLC code and displays it formatted "
                "to the user."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_keeps_configure_request_in_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="How can I configure nginx so requests to /api route to localhost:8001?",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_does_not_match_spec_inside_inspect(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Inspect the rust implementation of the algorithm and optimize it for performance.",
        )
        self.assertNotEqual(family, "tests")

    def test_classify_task_family_prefers_feature_over_docs_for_source_code_generation(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Please generate the source code for the repository in this directory. "
                "You can find the test cases and documentation, which you should adhere to and not modify."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_commit_implementation_request_with_readme_constraint_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Implement Commit 1 only for Pi support in /Users/prayagmatic/dev/pi-extensions/caveman.\n\n"
                "Scope:\n- Create Pi extension core.\n- Do not touch README.\n\n"
                "Requirements:\n"
                "1) Add extension file using Pi Extension API.\n"
                "2) Register /caveman [lite|full|ultra] and support deactivation via stop caveman.\n"
                "3) Keep changes minimal and clean."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_follow_on_commit_implementation_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Implement Commit 2 only for Pi support in /Users/prayagmatic/dev/pi-extensions/caveman.\n\n"
                "Dependency gate:\n"
                "- Before coding, wait until git log contains commit subject exactly: "
                "\"feat(pi): add caveman extension core modes\".\n\n"
                "Scope:\n"
                "- Extend Pi extension from Commit 1.\n"
                "- Add commands/modes for /caveman-commit and /caveman-review.\n\n"
                "Requirements:\n"
                "1) Use skills/caveman-commit/SKILL.md.\n"
                "2) Add mode persistence and config parity."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_live_update_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="I want to live update the currently detected type of input.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_video_quality_default_change_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See in the enhance setting on Media Lightbox for videos, can you make it so that the default "
                "quality is maximum and can you make sure that the quality thing is also shown in the upscale "
                "section of the shared task detail model thing?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_data_fetching_duplication_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look into this? Is this a valid concern around functionality or naming or what's the "
                "actual issue? Data Fetching Duplication (2,266 lines across 3 files). "
                "The problem: Developers don't know which hook to use. All three deal with generations but "
                "serve different views. What beautiful looks like: Two hooks max."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_query_key_consistency_cleanup_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                'Query key consistency: The "98% use queryKeys.*" claim is overstated. Found ~30 hardcoded '
                "query keys across 18 files. Actual compliance is closer to 80-85%. This is a good cleanup "
                "target - mechanical fixes, low risk. 4 down, 2 to go."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_supabase_discord_query_request_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Here's db credentials, can you look it up and share a very brif mesages on how to browse the "
                "discord_messages, users/channels names table to query this:curl "
                '"https://example.supabase.co/rest/v1/" -H "apikey: [REDACTED]"'
            ),
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_benchmark_bug_implementation_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "You are an expert Rust developer solving a task in Hyperswitch. "
                "TASK CONTEXT: - Repository: juspay/hyperswitch - Task ID: juspay__hyperswitch-9097 "
                "PROBLEM STATEMENT: Bug: Subscription Create to return client secret and create payment intent "
                "automatically. HINTS: Add Subscriptions API entrypoint to create a subscription intent."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_desloppify_scan_score_request_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you desloppify scan this repo using the live module and share the score and issues",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_subjective_code_paste_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at this code paste subjectively? Don't use the deslopify scanner "
                "and try to understand if it's holistically beautiful and well engineered?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_desloppify_scan_remediation_loop_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you run a scan on this repo and tackle all of the open issues? "
                "Don't stop until you've gotten the score as high as possible"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_open_desloppify_issue_scan_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see if there are open desloppify issues in this repo? "
                "Run it using this repo's own code"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_playtest_heavy_build_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Make a first person fast paced ultimate 3D shooter. Think CSgo but modern and ray-traced. "
                "Have lots of configs/settings, you will playtest the actual game, verify textures, sfx, "
                "and add it to this AI archive."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_set_up_deploy_script_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Set up a GitHub web deploy script for this.",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_run_repo_locally_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you run this please locally? https://github.com/xliry/banodoco-wrapped",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_run_repo_plz_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you run this repo plz",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_log_export_hook_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Hey! So I want to make sure that all of the logs from the conversations with Claude get saved "
                "to a generic repo on my computer which then gets pushed to Github. What's the best way to "
                "tackle this? Should it be an instruction that I use when launching the Claude interface?"
            ),
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_architecture_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "We've made a number of changes to this codebase quite rapidly. "
                "Please could we do an audit for architecture, code clarity, and maintainability?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_code_quality_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can we review our code quality and style in ping? You can check the snake extension.",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_video_item_usage_explanation_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What does VideoItem.tsx do - where/how is it used?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_run_scan_and_make_recommendations_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you run scan on this codebase and make reocmmendatiosn",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_desloppify_score_improvement_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What's our desloppify score now? How can we improve it? Run using this repo",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_desloppify_scan_to_target_score_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you run a desloppify scan on this repo and figure out how to get it to 95+?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_desloppify_scan_on_codebase_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you run a desloppify scan on this codebase?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typoed_deslopify_target_score_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you run deslopify on this repo and try to figure out the best possible way "
                "for us to get the score higher than 95?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_upgrade_my_code_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="upgrade my code to multiboot 2 so I can use UEFI and in turn GOP for Graphics",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_add_it_to_json_file_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="add it to models.json.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_bun_backend_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "write a bun backend that uses the databases available in ORV/*.db to serve data, "
                "make this backend under ./backend, use Elysia JS"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_pick_up_markdown_prompt_as_docs(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="lets pick up android.md",
        )
        self.assertEqual(family, "docs")

    def test_classify_task_family_treats_make_python_script_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "make a python script to get the Torah, Bible(KJV), Quran from Project Gutenberg "
                "and combine it all into a single abrahamic_corpus.txt"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_beauty_audit_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look through this code base from top to bottom and write a document where you kind of "
                "document things that are preventing the code from being beautiful. For example, inconsistent "
                "patterns, poor naming, and little details that are preventing it from being good and functional "
                "to being beautiful and amazing."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_beauty_audit_short_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What's preventing this codebase from being beautiful?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_repo_beauty_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="is this repo beautiful?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_database_security_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look through all the different database tables and try to understand that there's any "
                "that have security vulnerabilities or things that will cause issues for us? Be very diligent "
                "and compile a report for me please"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_legacy_debt_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look through this code base from top to bottom excluding the 1, 2, 2GP sub repo and "
                "try to identify places where the code is poorly structured or confusing or looks like it has "
                "legacy debt. Go through the overall structure and then each file one by one."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_todo_inventory_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Please can you inventory the @TODOs in the codebase and review each, "
                "to see if it is needed? Write your findings to a .md"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_hello_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="hello",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_load_command_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="/load",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_reply_with_just_the_number_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What is 2+2? Reply with just the number.",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_decruftify_scan_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you run @scripts/decruftify/ scan please",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_working_directory_question_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="what dir are you in",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_generate_some_text_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you generate some text please",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_current_project_question_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="what project are tyou in",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_npm_deploy_command_question_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="what would the CLI command be for this project if i wre to deploy it to NPM?",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_finish_where_you_left_off_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you finish where you left off?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_started_happening_since_extension_added_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="this started happening more since you added the session pruning extention, can you investigate it?",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_say_hello_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Say hello",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_on_delete_all_prompts_not_defined_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Technical details: onDeleteAllPrompts is not defined",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_image_gallery_screen_size_prompt_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see in the image gallery the way we're basing how many images to show on the "
                "dimensions of those images? Can you think of a way to do this that takes into account "
                "both that and the actual size of the screen?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_ipad_batch_mode_shot_images_layout_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "on ipad in batch mode, It only shows two images per line on the shot images editor so "
                "it looks very spaced out. Can you think about how to have it be more dynamic so that "
                "like you know gives the appropriate amount of images per line?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_single_purpose_edge_function_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "A single-purpose edge function that takes a task_id and returns its current status from "
                "the database using the service role (bypassing RLS). One query: SELECT status FROM tasks "
                "WHERE id = ?. Returns the status string."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_generation_pane_default_shot_exclude_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see in generation pane the way we have this when you're on your default shot, "
                "there's an exclude items with a position tag. However, that doesn't seem to be working. "
                "It seems to always show no images even when there are images."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_media_lightbox_mobile_video_poster_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See @src/shared/components/MediaLightbox/MediaLightbox.tsx On mobile and iPad videos "
                "don't display correctly particularly 916 ones high ones when the poster appears it "
                "basically shows like zoomed into the top part."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_media_lightbox_fill_images_black_border_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the fill images with AI functionality on the media lightbox. When I do that it seems "
                "to the image that's sent to the fill images with AI task has like a black outline border "
                "around it for some reason. Can you try to understand why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_decruftify_missing_areas_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at @scripts/decruftify/ and try to understand if there are missing areas? "
                "What kind of bad patterns or bad code or poor structure or problematic approaches do the "
                "various detectors and approaches here miss?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_batch_mode_reorder_empty_slot_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "When i switch the order of the first item in batch mode to later, it leaves the 0:0 slot "
                "empty, but really what it should do is shift all the items back proportionally so there's "
                "still an item at 0. can you figure out why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_single_image_timeline_duration_not_updating_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see when there's a single image on the timeline when, normally when I update the "
                "kind of duration inside the segment model, it updates the time in the timeline but when "
                "there's a single image it doesn't do this?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_structure_videos_showing_without_overlap_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "see the structure videos on @src/shared/components/SegmentSettingsForm/ - how do we "
                "determine when to pass/show them? They seem to be showing when the structure video "
                "doesn't overlap"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_final_video_section_show_video_instead_of_images_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the video we display in @src/tools/travel-between-images/components/FinalVideoSection.tsx "
                "- if a video exists there that wll be displayed for a shot can you also show it instead "
                "of the images?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_media_gallery_hide_edit_button_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "see @src/shared/components/MediaGallery/ can you not show the edit button on each item?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_media_gallery_delete_pagination_gap_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See on @src/shared/components/MediaGallery/ when i delete an item, it feels like it 'tried' "
                "to replace it with an item from the next page at the end of this page but it doesn't scuceed. "
                "Can you think of a smart way we should hadnle this so we never leave empty spots? "
                "This may be complicated,think it thorugh"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_tap_timeline_to_place_touch_device_message_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "when i select an item on timeline it shows 'tap timeline to place' "
                "but that should only show on ipads/touch devices"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_media_lightbox_load_images_button_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                'See the varnt selector on media lightbox? The way it shows "Load settings" for travel '
                'segments? If the image urls in those tasks params don\'t match the current ones, can you '
                'also show a "Load images" button that switches out the images? Think this through, for '
                "example, if they're a vairant of the current generaton, we should use that"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_media_lightbox_lineage_gif_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you see the lineage gif thing on @src/shared/components/MediaLightbox/ "
                "that shows when an open has more than 5 depth lineage?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_final_video_section_copy_button_success_state_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the share button on @src/tools/travel-between-images/components/FinalVideoSection.tsx "
                "when the copy button is there onn mobile it doesn't show a sucess state upon tp"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_final_video_section_drag_opens_lightbox_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See on @src/tools/travel-between-images/components/FinalVideoSection.tsx "
                "when I'm dragging over that secton, it frequently incorrectly opens the lightbox"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_segment_regenerate_frames_not_updating_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see when I change the frames on the segments generate form the number of frames "
                "per pair it doesn't seem to actually like update on the timeline or elsewhere can you "
                "try to understand why this is? "
                "@src/shared/components/MediaLightbox/components/SegmentRegenerateForm.tsx"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_upscaling_tracking_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See upscaling on @src/shared/components/MediaLightbox/ - that doesn't seem to track "
                "based on lke the other edit types do. Can you try to undersatnd why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_media_lightbox_arrow_keys_edit_mode_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "When I have a media lightbox opened, the left/right keys don't work when i'm in edit mode, "
                "only info mode. @src/shared/components/MediaLightbox/"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_generation_pane_mobile_view_shots_button_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see in generation pane the way we have a view this shot button or view my shot or "
                "it's see all images on mobile that only shows the icon and no text and it's also weirdly "
                "positioned when the exclude items with a position thing is visible. Can you make that into "
                "just say all shots or all images without the view and then make sure it's right next to the "
                "shot drop down and not shifted weirdly when the checkbox appears?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_release_readiness_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Alright, time for a new release. "
                "Check third party contributions since the last release, verify changelog coverage, "
                "and tell me if we are good to release."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_upload_integration_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I want to use https://github.com/badlogic/pi-share-hf "
                "to upload sessions on pi-nes to hf."
            ),
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_register_on_pypi_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you register this properly on PyPI",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_readme_link_update_issue_close_as_docs(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you update the discord link to here and comment on the issue with the link too and close it: "
                "https://discord.gg/aZdzbZrHaY"
            ),
        )
        self.assertEqual(family, "docs")

    def test_classify_task_family_treats_performance_speedup_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can we make this 3x faster please "
                "/Users/thomasmustier/Desktop/Screen Recording 2026-02-05 at 14.58.56.mov"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_ml_intern_sft_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="filter capybara to 100 samples and sft smollm2 on it. call the model yolo-mode",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_ml_intern_train_subset_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Train SmolLM3 on a subset of AgentTrove",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_claude_md_guided_product_implementation_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I'll start by setting up the project structure and creating a complete "
                "video player implementation following the CLAUDE.md guidelines."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_claude_md_guided_start_building_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I'll help you start building the RSS reader project following the CLAUDE.md "
                "operating mode. Let me first explore the current repository state, then create "
                "the necessary planning documents and begin parallel execution."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_claude_md_guided_bootstrap_project_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I'll help you bootstrap the love assistant project according to the CLAUDE.md "
                "operating mode. Let me start by exploring the current state of the repository "
                "and setting up the checkpoint infrastructure."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_comprehensive_plan_for_building_product_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I'll start by exploring the current state of the repository and then create a "
                "comprehensive plan for building the calculator tool with full parallel execution."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_context_files_query_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What context files are loaded? Just list paths.",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_push_everything_to_github_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="push everything to github",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_skills_md_elegance_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="See the SKILLS.md files, are they elegant and succint and well-explained? Any gaps?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_code_base_beautiful_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Is this code-base beautiful? What's preventing it from being beautiful?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_commit_authorship_question_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Is there any way to give this person commit authorship on all the work they did? "
                "https://github.com/peteromallet/desloppify/pull/103"
            ),
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_disk_cleanup_search_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you do a thorough search through the computer for any files that can be safely deleted? "
                "Just try to find like big folders that have lots of videos and images or accumulations of lots "
                "of small files."
            ),
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_free_up_space_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="I need to free up space on my computer, can you look around and come up with suggestions on what to do?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_cursor_junk_cleanup_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "We seem to have a lot of junk Cursor files everywhere. Can you find them all and come up with "
                "candidates for deletion? We don't use cursor anymore, so we can be very aggressive in terms of "
                "what we delete."
            ),
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_await_instructions_handoff_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Read the codebase context, then cd into /user_c042661f/.pm and await instructions",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_start_working_now_handoff_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Start working on the task now.",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_what_has_hf_released_lately_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What has HF released lately?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_folder_size_command_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="commande pour afficher la taille d un dossier",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_provider_context_file_query_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What provider-specific context file was loaded? Just the path.",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_read_justfile_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Read @justfile",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_plain_read_justfile_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="read justfile",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_list_files_in_directory_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="list files in this directory",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_review_and_explain_extension_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="We have an extension called long-task-harness. Please can you review the .ts and explain the behaviour",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_plugin_how_it_works_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="How does the claude code plugin work?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_tool_parameter_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="In @pi-interactive-subagents, what does the spawning parameter do?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_read_tool_behavior_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "How does the default read tool in Pi handle images? Specifically, I'd like to understand "
                "what happens when it encounters image files and whether there are special behaviors for "
                "images embedded in markdown."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_compare_extensions_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Want you to compare the following Pi extensions, "
                "- https://www.npmjs.com/package/@the-agency/pi-hashline-edit "
                "- https://github.com/RimuruW/pi-hashline-edit "
                "- https://github.com/Whamp/pi-read-map "
                "Please clone each of them in /tmp and explore their codebase before giving me your report."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_justfile_update_automation_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "This directory has multiple sub-directories where each repo is a cloned upstream repo of some pi extension. "
                "I want a justfile which has a `just update` command that fetches the latest changes in each repo and "
                "shows me a commit log / changelog of what was pulled."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_justfile_update_followup_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Update @justfile to show a summary of updates at the end. "
                "Useful in case it fails for any extension and I don't need to scroll back up to see what changed"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_justfile_update_prompt_timing_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Update the justfile to give me an update prompt after fetching changes right away. Not at the end",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_compare_subagent_extensions_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Compare @pi-interactive-subagents, @tintinweb-pi-subagents, and @pi-subagents .pi/",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_linking_subagents_architecture_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Should I be linking pi-tasks with pi-interactive-subagents somehow?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_researcher_subagent_demo_request_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Give me a demo of the researcher subagent",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_researcher_showcase_request_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="showcase the researcher agent",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_researcher_showcase_without_article_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="showcase a researcher agent",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_use_claude_tool_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="use claude tool",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_what_repo_is_this_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What repo is this?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_what_coding_agent_is_this_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What coding agent is this?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_pi_update_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="pi update",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_commit_and_push_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="commit work till now and push it",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_commit_changes_and_push_current_branch_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="commit the changes so far and push to current branch",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_shell_output_os_quiz_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Based only on the shell output above, what OS am I on? Answer in one word.",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_idle_sound_notification_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="I want claude to make a sound whenever it is waiting for my input or finished a task and is idle. Set this up for me.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_cancellation_exception_review_finding_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "providerInfo() uses runCatching { transport.isAvailable() }, which will also swallow "
                "CancellationException and prevent coroutine cancellation from propagating."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_subagent_codebase_bloat_research_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "For this task use subagents. I want to run a extensive research of the codebase and detect "
                "places where code has led to bloat."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_start_android_emulator_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="start android pixel9 emulator and run the android app on it",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_pair_android_emulators_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "a phone and a watch emulator both are running (android). I want you to pair them. "
                "Search the internet what is the best way to do this purely from cli using adb commands."
            ),
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_hf_cli_info_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="use the /hf-cli to get info about trending papers right now",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_default_model_settings_question_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="how can i change my default model in my Pi settings?",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_find_session_traces_in_directory_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you find the Pi sessions traces from this directory? how many are they? (in ~/.pi/agent/sessions)",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_find_buckets_related_code_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you find the buckets-related code in this repo",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_describe_status_line_of_pi_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="describe the status line of Pi (ie. the line at the bottom of the screen)",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_ra_diagnostics_on_main_rs_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="use ra_diagnostics on src/main.rs",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_extract_white_text_script_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "write a very fast script that extracts all white text in iclr2026_pdfs into "
                "iclr2026_pdfs_whittext.jsonl. flush results, and ensure that upon restarting the script "
                "already processed outputs are skipped. print progress"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_write_new_pdf_extraction_tool_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "write a new extraction tool based on src/extract_white_text.py that just extracts all "
                "text from pdfs into a jsonl. remove formatting or boxes"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_pi_share_hf_session_collection_request_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "regarding the pi-share-hf extension, can we not collect the session automatically when pi shutdown? "
                "Also, right now it seems like all sessions for a given date are put together, but I want each session "
                "to stay separate."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_remove_files_only_after_upload_success_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="re the script to update session data in ./sync-hf-sessions.sh: only remove files if upload command succeeded",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_generate_new_wallets_logic_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "make sure to generate new wallets for older challenges as long as possible, "
                "iff those have less difficulty than new challenges."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_rewrite_miner_logic_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "rewrite the miner logic like this: find the easiest eligible challenge and work on it. "
                "among the easiest, find the most recent one. look at challenges.json for examples on how challenges look like"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_em_dash_jsonl_search_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "use iclr2026_pdfs_text.jsonl to search for all papers that contain an excessive amount of em-dashes "
                "(maybe compute mean and stddev first and then check for >1 stddev)"
            ),
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_frontend_things_queue_overlap_prompt_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Few frontend things: Queue goes over the thinking text. With small width, we get display issues."
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_work_on_agents_tooling_inspection_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="I'd like to work on the agent's tooling. Please can we inspect where we are?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_where_are_you_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="where are you",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_pr_review_opinion_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you review this PR and the comments on it: https://github.com/gradio-app/gradio/pull/13508. Do you agree with suggestions or no?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_prettier_config_generation_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Generate a simple prettier config to format json/json(minfied) for nice viewing.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_link_harmonization_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Not a fan of how different are the session links when they show a title vs. when they don't. "
                "Harmonize them to have the title OR the hash thingy at the left and the date + time at the right."
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_short_extension_comparison_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="pi-link vs pi-mesh",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_component_interaction_explanation_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at how the shot images editor and the timeline work together? "
                "Like look at how they both for example move frames delete frames and so on "
                "and try to understand how it works."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_subjective_vs_mechanical_weighting_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="how are the subjective maesures weighted relative to the mechanical ones?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_subjective_re_evaluation_threshold_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Does next offer them to revavluate subjective iitems if tehy're <90?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_scan_narrative_subjective_rerun_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Does it menton in the scan narrative that they can rerun subjective analsis?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_multi_directory_scoring_feedback_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look into this feedback? What's the best way to run multiple directies? "
                "what would be sick for this is to be able to do per module/subdirectory score, "
                "for teams with different responsibilities, and also git ignore."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_latest_issue_solution_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look through the latest issue and try to understand if there's a good solution to it? "
                "Be really in depth. We obviously don't have a Windows machine. But is there a way to "
                "understand this in a way that will help solve it for everyone?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_language_specific_wiring_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Hey, can you look at all the things we do for specific languages in the forward slash langs "
                "folder and then try to identify if there's anything where we're not handling language specific "
                "functionality inside the language specific layer?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_pr_critical_assessment_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at the pull request that's on this repo and just assess it critically and "
                "understand whether it should be kept or ditched?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_no_tools_beauty_assessment_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you look at this codebase and try t understand if it;s beautiful? no tools, just your own assessment",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_without_tools_beauty_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="witjhout using delsoppify or any tools or Claude memory, answet the question: is this codebase beautiful?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_requirements_communication_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="do we communicate the requirements well?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_codebase_scoring_scheme_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Throughout the entire code base, we should just be using three scores to communicate the "
                "progress. The objective score, the subjective score and the strict score. Can you review "
                "that setup and simplify it?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_agent_instruction_succinctness_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "could the agent instructions be made more succinct? Why do we need two codex instructions? "
                "Be careful not to lose anything"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_command_surface_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you please look at all the commands and try to determine if they're if they're "
                "well-named and if there's a help command that reveals to the agent what are the commands "
                "available to it are and basically just try to determine whether the command surface is elegant."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_agent_instruction_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "could they agent instructons be made more succint? Like why do we need to two coex "
                "insttrucions? Be careful not to lose anything"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_prompt_nudging_policy_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you see the prompts we give to agents based on results from scans? Can you figure out "
                "when it makes sense to nudge them towards deploying multiple agents and when to nudge them "
                "towards using fixers that we have?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_issue_comments_state_of_play_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you look at the comments here and try to think of the state of play?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_issue_comments_state_of_play_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you look at the commments here and try to think of the state of play?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_score_review_and_improvement_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you check the desloppify score and figure out how to improve it",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_documents_desloppify_score_improvement_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you run documents/desloppify on this repo and figure out how to improve th score",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_documents_desloppify_strict_score_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you run documnets/desloppify from here and then try to get the strict score "
                "to as close as possible to 100."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_scan_score_llm_communication_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at how we communicate the scan score to the LLM? Can you print to the LLM and "
                "tell them that they should send it to the end user in a nice table?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_structure_md_philosophy_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Does STRUCTURE.md cover subjective and mechanical issues? Does it cover philosophy? "
                "Can you go through things, try to understand the philosophy and distill it for me "
                "into a short paragraph"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_subjective_detectors_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see the subjective detectors we have codebase wide? Where we search for "
                "non-explicit issues?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_any_usage_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you do an in-depth invesgigation into all the usages of 'any' in the code-base. "
                "We have 1900. Try to undersatdnd which are necessary and which are sloppy"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_subjective_detectors_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see the subjectivedetectors we have code-base wide? Where we search for "
                "non-explict issues?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_confused_issue_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you understand this - if they're confused, why are they?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_pr_subjective_questions_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at this PR and try to understand it and let me know what the C# specific "
                "questions, subjective questions, are asked of the codebase both at a file level and at "
                "a cross codebase level"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_open_issues_triage_and_close_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look through all the open issues and try to understand which of them are fixed, "
                "which should be ignored and which should be closed?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_unused_or_improperly_implemented_code_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you see if there's anything in the code that isn't being used or properly implemented?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_strict_score_improvement_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you try to understand how we can get the trict score for thsi repo closer to 100?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_desloppify_scan_audit_request_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you run a desloppify scan on this repo itself and let me know if you spot anything "
                "that needs to be included or needs to be improved?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_desloppify_scan_prompt_narrative_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you run a desloppify scan on this repo and just look at the actual stuff that we're "
                "sending to the lm, like the next command and whether we're presenting information in as "
                "nice a way as possible?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_subjective_question_review_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you critically assess the subjective questions that we ask of each codebase?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_language_subjective_questions_inventory_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="CAn you find the subjective questions we ask of each language? Can you see where htey're houses?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_beautiful_and_elegant_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "is this codebase beautiful and elegant? Deploy subagents to explore it across a number "
                "of dimensions to determine this"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_fundamentally_beautiful_and_elegant_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Is this fundamentally beautiful and elegant?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_holistically_beautiful_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="IS this codebase holistically beautiful?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_beautiful_and_elegant_dual_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="is this codebase beautiful? is it elegant?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_env_password_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you see if we have a db password in .env",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_push_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can pou push this here https://github.com/peteromallet/nigel.git",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_recent_git_commits_black_border_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at our recent Git commits? We made some changes to the fill edges with AI "
                "task in order to kind of remove a black border that was showing around the image. But "
                "that still seems to be showing."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_typo_heavy_refactoring_principles_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at the different component refacotring we've done today and make a general "
                "principles of refacoring and compoarmentalisation both for UX components and hooks?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_preset_not_applied_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See when i select a preset on @src/shared/components/SegmentSettingsForm/ that doesn't "
                "seem to actually use that phase config or pass the preset id to the task, can you see why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_temp_clone_compaction_prompt_inspection_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you temp clone this and find the prompt it uses to compact? https://github.com/anthropics/claude-code",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_duplicate_image_generation_path_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "why do we have @src/tools/image-generation/pages/ImageGenerationToolPage.tsx and "
                "@src/tools/image--generation/pages/ImageGenerationToolPage.tsx"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_deep_subjective_code_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you do a very deep subjective review of all the code in this repo? Try to look at things "
                "that the automated process would not discover and go very deep into them."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_warp_alt_enter_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I seem to not be able to use ALT+ENTER with Pi in Warp "
                "(even if I set left option to meta), yet option+up arrow works just fine in codex. "
                "Please can we investigate what's going on?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_explicit_bugfix_how_is_this_possible_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Bugfix: Jun 17 2026 have two pauper leagues. How is this possible?",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_layout_looks_weird_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "see scorecard.png, the layout of it looks weird now, can you try to understand what's "
                "happening with it, the spacing feels uneven"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_image_generation_optimistic_placeholder_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see the image generation thing when I am on the image generation form? Two problems. "
                "When I generate two batch of tasks with 10 each. There's this kind of optimistic placeholder "
                "thing that shows the task pane."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_code_explanation_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "waht does this code say?\n\n"
                "```ts\n"
                "// Submit (Enter)\n"
                "if (kb.matches(data, \"submit\")) {\n"
                "    if (this.disableSubmit) return;\n"
                "}\n"
                "```"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_bash_command_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="write a bash command to mv all files from iclr2026_pdfs into .",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_cursor_rules_location_query_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Where cursor rules are stored in my computer? example, the package management with uv one",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_ls_only_inventory_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="List what directories and files are here. Just ls, no explanation needed.",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_unzip_parallel_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="unzip all the files in this directory in parallel",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_eval_instance_filter_prompt_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Use the tools used in eval/figures/dllm_dingo.py to find those instances that fail "
                "when running constrained gsm8k but pass when running unconstrained. "
                "Also find those instances that are syntactically incorrect, constrained and do not have an autocompletion."
            ),
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_finish_script_adaptation_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Check out eval/figures/dllm_grammar_prompting.py. "
                "I started adapting it but please finish the job: it should show the results "
                "\"vanilla\" \"grammar prompting\" \"syntax\" \"auto\"."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_atomic_write_rewrite_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "rewrite miner.py such that files are only locked for writing "
                "and writes are done atomically (write to tmpfile then move to new location)"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_browser_mining_automation_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Check out https://sm.midnight.gd/wizard/mine. "
                "Can you write a script, e.g. Playwright, that automatically starts several mining sessions in parallel?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_parallel_launcher_script_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Write a very simple script that launches n browser_mining.py in parallel and prints out the combined logs.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_browser_mining_403_prompt_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I get 403 when running browser_mining.py after clicking sign. "
                "Can you check what is happening? It works in my normal browser."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_broken_edit_page_controls_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "@src/tools/edit-images/pages/EditImagesPage.tsx In the edit images page "
                "the selectors to click into different tools don't seem to be working or "
                "none of the forms seem to be inputable for some reason can you understand why"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_reference_section_loading_glitch_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can we see the reference section on the image generation form? Like that's loading, "
                "it seems to show the right skeletons, but then it seems to show a lot of images that "
                "shouldn't be there or maybe bigger than they should be. And then it goes back down to "
                "the kind of like right level. Can you see if just a weirdness happening when the "
                "references are loading? @src/tools/image-generation/components/ImageGenerationForm/ "
                "or maybe complication?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_share_button_failure_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "The share button on @src/tools/travel-between-images/pages/VideoTravelToolPage.tsx "
                "isn't working, can you put logs to understand why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_timeline_duplicate_skeleton_glitch_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See when i duplicate an item on @src/shared/hooks/timeline/ that's between this and "
                "other item (not the end item),the skeleton shows perfectly in between the duplicated "
                "item and the next. But then the actual image shows further ahead and the skeleton "
                "sticks around."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_lightbox_ipad_positioning_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Is tehre a known issue around opening lightboxes on ipad where it causes positoning "
                "weirdness with th elements below? LIke @src/shared/components/MediaLightbox/ for example"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_default_prompt_tag_misplacement_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see on video Travel Tool page on the segment controls segment on the video on "
                "the video regeneration controls I think it's called the model we show a default tag on "
                "the prompt and negative prompt fields when we shouldn't?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_connection_loading_jump_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See when i load the websiet, if the connecton isn't snappy, it first shows a screen "
                "and then the video or placeholder image jumps in. I'd like to make this nicer."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_image_jump_to_shot_context_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the way with travel segments we jump into the shot context when clicked via "
                "@src/shared/components/TasksPane/? I want to do a similar thing with images - if "
                "they belong to a shot via shot_id, to jump in and view that shot."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_quick_win_list_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you look into the both of these and execute them if they seem sensible? Quick wins "
                "(1-2 days): 1. SharedMetadataDetails.tsx - just extend the interface, 28 casts "
                "disappear 2. Remove dead shouldSkipCount code in use..."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_default_valjohn_display_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See @src/shared/components/SegmentSettingsForm.tsx The way we show the default "
                "Valjohn things - see in @src/shared/components/MediaLightbox/MediaLightbox.tsx For "
                "images in move mode we show a default prompt but we have..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_medialightbox_restructure_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "@src/shared/components/MediaLightbox/ is excessively large, how could it be "
                "restrucured in order to reduce the line count and complexity of the main component?"
            ),
        )
        self.assertEqual(family, "refactor")

    def test_classify_task_family_treats_ipad_structure_video_endpoint_request_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the structure video thing on the "
                "@src/tools/travel-between-images/components/Timeline/ - where you positon the "
                "structure video? On ipad, can you make it grab an end point by tapping and "
                "release or move it by tapping?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_before_after_textfield_scroll_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See in @src/shared/components/SegmentSettingsForm.tsx can you make it so i can't "
                "scroll inside the 'before' and 'afeer' text fields"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_arrow_keys_jumping_out_of_text_field_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see on Media Lightbox when I am in a text field for example on the inpaint "
                "or the text field when I'm writing inside of it and I press the left slash right "
                "keys on the keyboard it jumps to the next or previous..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_trailing_video_visibility_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you see the trailing video thing on timeline? whe the final item has a segment "
                "video, can you make it so it always shows it - even if the user hasn't added a "
                "trailing frame?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_video_upload_button_overflow_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "see the generate form on medialightbox? In advanced, mdoe, when space is "
                "contrasint, the replace/browse button for the video upload flows over the right, "
                "could you make it instead just show the icons for those buttons?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_escape_key_close_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Pressing escape key to clsoe the media lightbox stopped working @src/shared/components/MediaLightbox/",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_floating_shot_selector_ipad_overlap_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the floating header on the video travel tool page? The one that the floating "
                "one. I mean the floating shot selector that works perfectly on desktop but on iPad "
                "sizes it ends up behind the header."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_image_generation_modal_missing_backdrop_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "on @src/tools/travel-between-images/pages/VideoTravelToolPage.tsx See when i open "
                "the image generaton modal via this - why does it not show the blac background thing "
                "whereas it does when i open via the pane control ta..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_variant_count_edit_mode_mismatch_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see on the Media Lightbox the way when we're in info mode we show a variant "
                "count and button that points down to the variants? That shows an info mode for both "
                "videos and images but then when we go into edit m..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_move_tool_save_hang_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "when i press save on the 'move' tool for images in medialightbox, it hangs "
                "indefinitely. can you try to understand why? Put logs if need be to get to the "
                "bottom of it"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_locked_task_pane_lightbox_overlap_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see why sometimes when i open a medialightbox and the task pane is locked, "
                "the lightbox doesn't account for the task pane ebing locked and appears behind it "
                "instead of adjusted by it?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_ipad_single_image_selection_weirdness_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see on iPad with the timeline? When I tap a select a single image the tap "
                "timeline in place thing doesn't show it only shows when I select two images and then "
                "there's generally some weirdness around selecting..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_edit_video_tool_selector_missing_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "@src/tools/edit-video/pages/EditVideoPage.tsx See on edit video page, the tool "
                "selector isn't showing just goes right into a single tool. Everything works fine "
                "when I view the same form via the media light box,..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_ai_prompt_output_drift_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "@supabase/functions/ai-prompt/ Can you see why, when I said, quote, and 'the camera "
                "flies through the sky to the distant hills as the snow storm begins' The prompts "
                "that was generated was a lot more poetic and came ou..."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_timeline_yellow_notice_overlap_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you also see on the timeline when I start dragging an item these kind of like "
                "yellow time notices appear? And yeah can you just remove them they kind of overlap "
                "with the other notices that we already have."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_segment_settings_hide_fields_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "see in @src/shared/components/SegmentSettingsForm.tsx - can you not show these "
                "fields: Model i2v lightning baseline 2 2 2 Resolution 768x576 Seed Ran"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_medialightbox_regenerate_button_failure_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See on @src/shared/components/MediaLightbox/ the regenerate button doens't seem to "
                "be working, can you try to undersatnd why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_media_lightbox_edit_button_removal_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see the edit button that shows in the top left of the media light box Can "
                "you remove that on both desktop and mobile it shows on images?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_dead_code_question_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="see @src/shared/components/ShotImageManager/MobileImageItem.tsx is this dead code?",
        )
        self.assertEqual(family, "refactor")

    def test_classify_task_family_treats_documents_desloppify_typo_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you run documents/desloppify on thsi repo and try to get to as close to 100% strict score as possible",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_latest_documents_desloppify_strict_score_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you run the latest documents/desloppify on this repo adn share the strict score",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_pr_visibility_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you see the PR we have on this repo?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_frontend_relaunch_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you please re-launch the front end",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_downloaded_file_dimension_comparison_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you look at the dimensions of the last two downloaded files and let me know what the difference between them is?",
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_vertical_video_chop_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you chop the last vdieo uploaded in two vertically?",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_side_by_side_video_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you put those last 2 downloaded videos side by side?",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_voice_box_layout_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see how the text feels the way we have the voice box and the X? I want to "
                "make it so that the voice thing takes, if there's more, if the space it's on the "
                "right side and it takes up two rows..."
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_generation_slider_width_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see The generation slider and the variant toggle on the MediaLite box. Can "
                "you basically make the two of them take up 80% of the width instead of 100%?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_decruftify_structure_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look at @scripts/decruftify/ - Can you try to understand that the structure "
                "of this as a whole is good? Like try to figure out like how could this itself be as "
                "good as possible?"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_simple_realtime_manager_typing_recommendation_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you look into the below, understand its purpose, and make a recommendaton? 3. "
                "Type SimpleRealtimeManager.ts with Supabase payload types"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_name_review_prompt_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What's a better name for this? Desloppify/",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_audio_resize_for_doc_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you make head_swap_compare_00018-audio.mp4 small enough so it works in a doc?",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_image_fade_video_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you see the last two downloaded images and videos can can you make a little "
                "video where it basically starts with the image example image one then quickly "
                "afterwards example image two fades in..."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_ransom_security_incident_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "A hacker told us that he has access to all of the data to do with the app but I "
                "can't tell how he got it or what he did or if he's bullshitting. He said he's going "
                "to leak it all if we don't pay him a ransom. Can you..."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_table_vulnerability_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you look through all the different tables that we're using for this app to try "
                "to understand if there's any vulnerabilities or anything concerning with regards to "
                "how we're using and displaying or updating or dele..."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_optimistic_submit_workflow_review_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you look through all the different kinds of edit tasks for videos and images "
                "and look at how the video regenerate type on media light box look at how it "
                "basically optimistically submits the task then shows a place..."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_playwright_browser_mining_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Check out https://sm.midnight.gd/wizard/mine. "
                "Can you write a (e.g. Playwright?) script that automatically starts several mining sessions in parallel?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_api_endpoint_extraction_prompt_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Download the atconf app and extract all API endpoints that it fetches data from.",
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_cross_repo_loc_count_request_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you count all the lines of code I've done over the past three months across every repo?",
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_patch_version_bump_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="bump the patch version",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_bump_up_the_version_prompt_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you bump up the version of this",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_push_to_github_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="push to github",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_most_recent_issue_lookup_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you look at the most recent issue and see the most recent one",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_issue_still_happening_question_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="IS this issue stll happening? https://github.com/peteromallet/desloppify/issues/111",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_git_commit_summary_request_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you look at all the github commits since Sunday first thing and summarise everyhing "
                "that was done. Be thorough but succint"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_unpushed_changes_check_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you see if we have a bunch of unpushed changes?",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_apk_endpoint_inspection_as_data_analysis(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="inspect the apk in this directory for used API endpoints",
        )
        self.assertEqual(family, "data-analysis")

    def test_classify_task_family_treats_remove_search_from_page_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Remove the search from the page.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_tooltip_hover_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Display the discovered evidence as a tooltip view when hovering the detected language button.",
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_timeline_notification_question_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see how we show the pending notification thing on the video output strip on the timeline? "
                "@src/tools/travel-between-images/components/Timeline/SegmentOutputStrip.tsx "
                "@src/tools/travel-between-images/components/Timeline/"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_structure_video_setting_hidden_behind_action_bar_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See on @src/tools/travel-between-images/components/Timeline/ The structure video thing. "
                "See the one-to-one mapping versus fit to range setting on it. When I select an item "
                "it seems that that field moves behind the action bar."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_structure_video_setting_hidden_behind_actual_timeline_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See on @src/tools/travel-between-images/components/Timeline/ The structure video thing. "
                "See the one-to-one mapping versus fit to range setting on it. When I select an item "
                "it seems that that field moves behind the actual timeline so it's no longer viewable."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_default_enhance_prompt_to_false_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see on @src/shared/components/MediaLightbox/ regenerate form? "
                "@src/shared/components/SegmentSettingsForm/? Can you default Enhance Prompt to false?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_gallery_padding_bottom_error_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Why on on when i click into the gallery on "
                "@src/tools/image-generation/pages/ImageGenerationToolPage.tsx or "
                "@src/tools/travel-between-images/pages/VideoTravelToolPage.tsx do i get this: "
                "technical details: co.paddingbottom is not a function."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_parent_generation_assignment_question_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you see why this task got set to the parent generaton too?",
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_issue_queue_investigation_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you see this GitHub issue? I think that's surfacing a real problem where issues "
                "don't make it into the next queue. I think they probably should be very high priority, "
                "maybe at the top of the list. Can you look into this and try to understand what's "
                "happening https://github.com/peteromallet/desloppify/issues/117"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_mobile_advanced_mode_overflow_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "On mobile, the @src/shared/components/SegmentSettingsForm.tsx the advanced mode "
                "container causes things to run over the right side - it feels unevenly weighted "
                "towards the right"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_project_how_it_works_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="how does this project work",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_delightful_ux_ideation_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Read to get context, but don't touch any code. I'm trying to think what would make "
                "the experience here delightful."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_terminal_status_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I'd like to be able to see in the terminal tab status when Pi is done replying. "
                "Do we have anything that would let us do this?"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_space_invaders_extension_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="We have the snake extension. Can we make a space invaders one",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_any_usage_streamlining_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Is there any way for you to categorize all the any usage to try to understand if "
                "and how it can be streamlined? Some of it is obviously necessary but obviously "
                "some of it isn't."
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_loading_screen_design_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="I'd like to have a nicer loading screen for articles. Can we design this?",
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_arcade_roundout_ideation_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "We have extensions for /snake, /space-invaders, /ping (pong). "
                "What are we missing to round out our arcade? Probably tetris"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_arcade_roundout_mario_variant_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "We have extensions for /snake, /space-invaders, /ping (pong). "
                "What are we missing to round out our arcade? Probably Mario"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_render_loop_log_investigation_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you try to understand why we get all these render loop logs when we click into "
                "a shot? Is this a legitimate concern or is it just some area where we have "
                "overzealous logging?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_section_navigation_tweak_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I'd like to make a couple tweaks - Allow left/right to navigate between sections "
                "when you reach the first/last page in the section"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_stale_article_date_report_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I am only seeing articles from april 1st when I runt he economist today "
                "(april 1 was yesterday). Do we know why? Are we not updating?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_economist_demo_layout_and_export_regression_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Couple things with economist demo - article lists for 2 sections are indented "
                "differently to other sections - Middle East & Africa - Science & Technology "
                "Please can we investigate why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_media_lightbox_shift_on_ipad_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "On the Media Lightbox a weird thing happens where it feels like the Media "
                "Lightbox kind of on iPad when I click into a field or something it feels like "
                "the Media Lightbox kind of shifts weirdly"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_doom_extension_ideation_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="What would it take to get Doom as a pi extension",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_travel_segment_navigation_request_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See when I clcik into a travel segment from within the @src/shared/components/TasksPane/, "
                "I want it to open within the context of the shot it's a part of - similar to how it would "
                "when i open via @src/tools/travel-between-images/components/Timeline/SegmentOutputStrip.tsx. "
                "For example, so that i can click back/forth to previous segments, etc. "
                "Can you understand what i need to do this?"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_component_structure_audit_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Is the entire @src/tools/travel-between-images/components/Timeline.tsx component impeccably structured?",
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_page_dropdown_difference_question_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See the page dropdown in @src/shared/components/TasksPane/ - why is that different to the "
                "page dropdown at the top of @src/shared/components/MediaGallery/ in "
                "@src/tools/image-generation/pages/ImageGenerationToolPage.tsx"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_complete_task_upscale_fix_plan_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Implement the following plan: Fix image-upscale Variant Creation. "
                "Problem Summary: image-upscale tasks create variants via WRONG handler. "
                "Actual: variant_type = 'edit'. Expected: variant_type = 'upscaled'. "
                "Root Cause: handleVariantCreation is called for upscale. "
                "Solution: Minimal Fix. Modify supabase/functions/complete_task/index.ts and deploy complete_task."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_clone_repo_into_wrapped_page_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you please get clone the following repo and then make it into a page that lives at "
                "forward slash wrapped and um, and, um, yeah, make, get everything working properly at that "
                "page. Um, the extremely robust and thorough. https://github.com/xliry/banodoco-wrapped"
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_web_asset_weight_reduction_prompt_as_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you git clone this https://github.com/banodoco/corner-play-dance and try to figure out "
                "if you can make the assets that show on the page less heavy for example they're probably "
                "unoptimized for the web right now look into it"
            ),
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_treats_lightbox_edge_draw_close_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See @src/shared/components/MediaLightbox/hooks/inpainting/ on "
                "@src/shared/components/MediaLightbox/ - when i draw over the edge and release it closes "
                "the lightbox, but it should block closing and continue drawing at the edges. "
                "can you understadn why?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_oversized_hooks_inventory_request_as_planning(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Oversized hooks (7 files >800 lines, e.g., useShots.ts at 2,350 lines) - "
                "can you investigate what these are? Make a list"
            ),
        )
        self.assertEqual(family, "planning")

    def test_classify_task_family_treats_typo_heavy_push_everything_request_as_control(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="PUSHE VERYTHING TO GITHUB",
        )
        self.assertEqual(family, "control")

    def test_classify_task_family_treats_supabase_function_deploy_request_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you try to deploy @supabase/functions/complete_task/?",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_video_transition_anomaly_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "See 'Preview all generated segments together' On video travel tool page, for some reason "
                "the transitions between the videos are all smooth apart from the transition from video one "
                "to video two. Can you understand because I'm thinking unusual about this one relative to the others?"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_timeline_delete_shift_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "When i delete a timeline item, it shifts the video "
                "@src/tools/travel-between-images/components/Timeline/SegmentOutputStrip.tsx but leaves an "
                "extra item in the next position that then disappears. similarly weird things happen when I move them around."
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_ipad_taskspane_unlock_close_behavior_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Can you try to understand why - wheni unlock the @src/shared/components/TasksPane/TasksPane.tsx "
                "on ipad - the page content adjusts but then the pane only closes when i tap outside"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_variant_selector_missing_in_edit_mode_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "can you see why the varant selector isn't showing on "
                "@src/shared/components/MediaLightbox/ in edit mode? Either for videos or images"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_structure_video_empty_state_issue_as_bugfix(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "i deleted all the structure videos on the timeline, but still getting this empty state "
                "and every segment is showing a structure video"
            ),
        )
        self.assertEqual(family, "bugfix")

    def test_classify_task_family_treats_nginx_path_rewrite_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="In nginx.conf how can I remove the /api part of the request for forwarded calls?",
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_pdf_scanner_script_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Please write a simple python script to scan pdf for prompt injections. "
                "It should look for white text and similar things."
            ),
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_adapt_download_script_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Can you adapt the ICLR download script to download desk-rejected submissions?",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_adapt_code_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Please adapt the code to fetch all eligible challenges from the API first.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_benchmark_flamegraph_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Adapt the benchmark to allow quickly iterating on results, then use flamegraphs to identify hotspots and optimize them away.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_github_workflow_dispatch_prompt_as_setup_build(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "Look at .github/workflows/release.yml. I want to enable workflow_dispatch manually "
                "and fetch upload_url from the GitHub API."
            ),
        )
        self.assertEqual(family, "setup-build")

    def test_classify_task_family_treats_move_function_into_rust_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Please move the entire function mine_challenge_native into the rust code for faster processing.",
        )
        self.assertEqual(family, "refactor")

    def test_classify_task_family_treats_dependency_swap_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Please adapt this code to use the argonautica package instead of the cryptoxide variant.",
        )
        self.assertEqual(family, "refactor")

    def test_classify_task_family_treats_restructure_component_prompt_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview=(
                "I want to restructure PhaseConfigSelectorModal to reduce the line count and simplify it. "
                "Can you look at how we structure Timeline and figure out how we can do the same?"
            ),
        )
        self.assertEqual(family, "refactor")

    def test_classify_task_family_treats_find_and_remove_shims_prompt_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="can you go through the codebase and find/remove any shims? Update tests if need be",
        )
        self.assertEqual(family, "refactor")

    def test_classify_task_family_treats_script_extension_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Adapt the donate_all_wallets script to also work for the .wallet data in wallets/.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_code_and_compare_algorithm_prompt_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Code suffix array construction in Python. Compare it with a brute-force approach.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_coordination_script_request_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Write a coordination script similar to launch_miner for browser_mining.py.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_treats_resubmit_solution_adaptation_as_feature(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Adapt resubmit_solutions by trying to register the address before submitting the solution.",
        )
        self.assertEqual(family, "feature")

    def test_classify_task_family_does_not_treat_ui_replacement_phrase_as_refactor(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Use read-book-icon.svg instead of the book emoji for the documentation on the main page.",
        )
        self.assertNotEqual(family, "refactor")

    def test_classify_task_family_detects_frontend_ui(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Make it use the PlusJakartaSans font and restyle the React landing page.",
        )
        self.assertEqual(family, "frontend-ui")

    def test_classify_task_family_does_not_match_ui_inside_unrelated_word(self) -> None:
        family = review.classify_task_family(
            task="",
            prompt_preview="Write a script to scrape arXiv papers while staying compliant with arXiv guidelines.",
        )
        self.assertNotEqual(family, "frontend-ui")

    def test_build_review_aggregates_recommendations(self) -> None:
        payload = review.build_review(
            data_root=Path("."),
            datasets=[],
            max_rows=10,
            examples_per_family=1,
        )
        self.assertEqual(payload["aggregate_task_family_counts"], {})
        self.assertEqual(payload["top_recommendations"], [])


@unittest.skipIf(pa is None or pq is None, "pyarrow not installed")
class TrajectoryTaskReviewDatasetTests(unittest.TestCase):

    def test_review_dataset_summarizes_task_families_and_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_root = data_root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "messages": [
                        [
                            '{"role":"user","content":"Fix the failing parser and rerun the tests."}',
                            '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":"{\\"path\\": \\"src/parser.py\\"}"}}]}',
                            '{"role":"tool","name":"read_file","content":"def parse(x):\\n    return x"}',
                            '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"run_test","arguments":"{\\"command\\": \\"pytest -q tests/test_parser.py\\"}"}}]}',
                            '{"role":"tool","name":"run_test","content":"FAILED tests/test_parser.py::test_parse\\nE   assert 1 == 2"}',
                        ],
                        [
                            '{"role":"user","content":"Make it use the PlusJakartaSans font and override the default system font."}',
                            '{"role":"assistant","content":"","tool_calls":[{"function":{"name":"str_replace_editor","arguments":"{\\"path\\": \\"src/theme.css\\"}"}}]}',
                            '{"role":"tool","name":"str_replace_editor","content":"ok"}',
                        ],
                    ]
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = review.review_dataset(
                data_root,
                "trace-commons-agent-traces",
                max_rows=10,
                examples_per_family=2,
            )

        self.assertEqual(summary["dataset"], "trace-commons-agent-traces")
        self.assertEqual(summary["task_family_counts"]["bugfix"], 1)
        self.assertEqual(summary["task_family_counts"]["frontend-ui"], 1)
        self.assertTrue(any("validation" in flag for flag in summary["insight_flags"]))
        self.assertEqual(summary["task_family_examples"]["bugfix"][0], "Fix the failing parser and rerun the tests.")

    def test_review_dataset_uses_terminalbench_task_name_instead_of_warmup_example(self) -> None:
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_root = data_root / "terminalbench-trajectories" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "task_name": ["adaptive-rejection-sampler"],
                    "steps": [
                        [
                            {"src": "user", "msg": "Warmup"},
                            {"src": "agent", "msg": "I will inspect the repository and fix the issue."},
                            {"src": "tool", "tool_name": "run_shell", "msg": "ls"},
                            {"src": "tool", "tool_name": "edit", "msg": "patched"},
                        ]
                    ],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = review.review_dataset(
                data_root,
                "terminalbench-trajectories",
                max_rows=10,
                examples_per_family=2,
            )

        self.assertEqual(summary["task_family_examples"]["other"][0], "adaptive-rejection-sampler")

    def test_review_dataset_dedupes_displayed_examples_after_truncation(self) -> None:
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_root = data_root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            shared_prefix = "Implement the React dashboard with a responsive layout and token-aware search. " * 4
            prompt_a = shared_prefix + "Variant A"
            prompt_b = shared_prefix + "Variant B"
            table = pa.table(
                {
                    "prompt": [prompt_a, prompt_b],
                    "messages": [
                        [json.dumps({"role": "user", "content": prompt_a}), json.dumps({"role": "assistant", "content": "I will inspect the repo."})],
                        [json.dumps({"role": "user", "content": prompt_b}), json.dumps({"role": "assistant", "content": "I will inspect the repo."})],
                    ],
                    "harness": ["claude-code", "claude-code"],
                    "session_id": ["s1", "s2"],
                    "tools": [json.dumps([]), json.dumps([])],
                    "metadata": [json.dumps({}), json.dumps({})],
                    "sent_at": ["", ""],
                    "num_user_messages": [1, 1],
                    "num_tool_calls": [0, 0],
                    "trace": ["", ""],
                    "file_path": ["data/train-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = review.review_dataset(
                data_root,
                "trace-commons-agent-traces",
                max_rows=10,
                examples_per_family=2,
            )

        self.assertEqual(len(summary["task_family_examples"]["frontend-ui"]), 1)

    def test_review_dataset_rows_scanned_matches_reviewed_trace_commons_rows_with_trace_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_root = data_root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            table = pa.table(
                {
                    "prompt": [None],
                    "messages": [json.dumps([])],
                    "harness": [None],
                    "session_id": ["s1"],
                    "tools": [json.dumps([])],
                    "metadata": [json.dumps({"trace_type": "structured"})],
                    "sent_at": [""],
                    "num_user_messages": [0],
                    "num_tool_calls": [0],
                    "trace": [
                        json.dumps(
                            {
                                "messages": [
                                    {
                                        "info": {"role": "user"},
                                        "parts": [{"type": "text", "text": "Fix the broken parser and rerun tests."}],
                                    }
                                ]
                            }
                        )
                    ],
                    "file_path": ["data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = review.review_dataset(
                data_root,
                "trace-commons-agent-traces",
                max_rows=10,
                examples_per_family=2,
            )

        self.assertEqual(summary["rows_scanned"], 1)
        self.assertEqual(summary["task_family_counts"]["bugfix"], 1)

    def test_review_dataset_classifies_from_full_prompt_not_truncated_preview(self) -> None:
        with TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            dataset_root = data_root / "trace-commons-agent-traces" / "data"
            dataset_root.mkdir(parents=True)
            long_prefix = "Token budget notes. " * 30
            prompt = long_prefix + "Make the React landing page use PlusJakartaSans and restyle the UI."
            table = pa.table(
                {
                    "prompt": [prompt],
                    "messages": [[json.dumps({"role": "user", "content": prompt})]],
                    "harness": ["claude-code"],
                    "session_id": ["s1"],
                    "tools": [json.dumps([])],
                    "metadata": [json.dumps({})],
                    "sent_at": [""],
                    "num_user_messages": [1],
                    "num_tool_calls": [0],
                    "trace": [""],
                    "file_path": ["data/train-00000-of-00001.parquet"],
                }
            )
            pq.write_table(table, dataset_root / "train-00000-of-00001.parquet")

            summary = review.review_dataset(
                data_root,
                "trace-commons-agent-traces",
                max_rows=10,
                examples_per_family=2,
            )

        self.assertEqual(summary["task_family_counts"]["frontend-ui"], 1)


if __name__ == "__main__":
    unittest.main()
