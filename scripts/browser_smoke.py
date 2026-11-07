#!/usr/bin/env python3
"""Repeatable Chrome + WebKit responsive/touch smoke test for the dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("CLICKER_TEST_URL", "http://127.0.0.1:8080")
WIDTHS = (320, 390, 431, 768, 1024)
ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "scratch" / "browser"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def assert_layout(page, width: int) -> None:
    overflow = page.evaluate(
        "Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - innerWidth"
    )
    assert overflow <= 1, f"{width}px viewport has {overflow}px horizontal overflow"
    if width <= 768:
        nav = page.locator(".bottom-nav")
        assert nav.is_visible(), f"bottom navigation hidden at {width}px"
        items = nav.locator("button, a")
        assert items.count() == 6
        boxes = [items.nth(index).bounding_box() for index in range(6)]
        assert all(box and box["height"] >= 44 for box in boxes), boxes
        for left, right in zip(boxes, boxes[1:]):
            assert left["x"] + left["width"] <= right["x"] + 1, boxes
    else:
        assert page.locator(".topbar .tabs").is_visible()


def assert_help_page(context, width: int) -> None:
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{BASE_URL}/help", wait_until="domcontentloaded")
    page.locator("[data-help-page]").wait_for()
    assert page.locator("#legacy").is_visible()
    assert page.locator("#training").count() == 1
    assert "train_commercial_model.py" in page.locator("#training").inner_text()
    overflow = page.evaluate(
        "Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - innerWidth"
    )
    assert overflow <= 1, f"help page at {width}px has {overflow}px horizontal overflow"
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOTS / f"help-{width}.png"), full_page=True)
    assert not errors, errors
    page.close()


def assert_wizard_capture_feedback(page) -> None:
    selected = []
    devices = {
        "devices": [
            {"id": "guermok-id", "name": "Guermok USB3 Video"},
            {"id": "facetime-id", "name": "FaceTime HD Camera"},
        ],
        "selected": "USB3.0 Video",
    }
    response = {
        "found": True,
        "frame": True,
        "device_name": "Guermok USB3 Video",
        "index": 2,
        "resolution": "1920x1080",
    }

    def handle_selection(route):
        selected.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body='{"ok":true}')

    page.route(
        "**/api/devices/capture",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(devices)
        ),
    )
    page.route("**/api/devices/capture/select", handle_selection)
    page.route(
        "**/api/capture-check",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(response)
        ),
    )
    page.evaluate(
        """() => {
            wizardStep = 0;
            renderWizard();
            document.querySelector('#setup-wizard').classList.remove('hidden');
        }"""
    )
    page.wait_for_function(
        """() => {
            const select = document.querySelector('#wizard-capture-select');
            return select && !select.disabled && select.options.length === 2;
        }"""
    )
    page.locator("#wizard-capture-select").select_option("guermok-id")
    page.locator("#wizard-capture-check").click()
    page.locator("#wizard-capture-status.success").wait_for()
    assert "Guermok USB3 Video" in page.locator("#wizard-capture-status").inner_text()
    assert selected == [{"id": "guermok-id", "name": "Guermok USB3 Video"}], selected
    page.locator("#setup-wizard").evaluate("element => element.classList.add('hidden')")
    page.unroute("**/api/devices/capture")
    page.unroute("**/api/devices/capture/select")
    page.unroute("**/api/capture-check")


def assert_wizard_appletv_pairing(page) -> None:
    started = []
    finished = []
    devices = {
        "devices": [
            {
                "id": "living-room-id",
                "name": "Living Room",
                "address": "192.0.2.10",
                "services": ["Protocol.Companion", "Protocol.AirPlay"],
            }
        ],
        "selected": None,
    }

    def handle_start(route):
        body = route.request.post_data_json
        started.append(body)
        response = {
            "token": body["protocol"].lower() + "-token",
            "protocol": body["protocol"],
            "device_provides_pin": True,
            "pin_code": None,
        }
        route.fulfill(status=200, content_type="application/json", body=json.dumps(response))

    def handle_finish(route):
        finished.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body='{"paired":true}')

    page.route(
        "**/api/devices/appletv",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(devices)
        ),
    )
    page.route("**/api/devices/appletv/pair/start", handle_start)
    page.route("**/api/devices/appletv/pair/finish", handle_finish)
    page.evaluate(
        """() => {
            wizardStep = 1;
            renderWizard();
            document.querySelector('#setup-wizard').classList.remove('hidden');
        }"""
    )
    page.wait_for_function(
        """() => {
            const select = document.querySelector('#wizard-appletv-select');
            return select && !select.disabled && select.value === 'living-room-id';
        }"""
    )
    page.locator("#wizard-auth-start").click()
    page.locator("#wizard-auth-pin").fill("1234")
    page.locator("#wizard-auth-finish").click()
    page.wait_for_timeout(250)
    auth_status = page.locator("#wizard-auth-status")
    assert "success" in (auth_status.get_attribute("class") or ""), (
        auth_status.inner_text(), started, finished
    )
    page.locator("#wizard-airplay-start").click()
    page.locator("#wizard-airplay-pin").fill("5678")
    page.locator("#wizard-airplay-finish").click()
    page.wait_for_timeout(250)
    airplay_status = page.locator("#wizard-airplay-status")
    assert "success" in (airplay_status.get_attribute("class") or ""), (
        airplay_status.inner_text(), started, finished
    )
    assert started == [
        {"identifier": "living-room-id", "protocol": "Companion"},
        {"identifier": "living-room-id", "protocol": "AirPlay"},
    ]
    assert finished == [
        {"token": "companion-token", "pin": 1234},
        {"token": "airplay-token", "pin": 5678},
    ]
    page.locator("#setup-wizard").evaluate("element => element.classList.add('hidden')")
    page.unroute("**/api/devices/appletv")
    page.unroute("**/api/devices/appletv/pair/start")
    page.unroute("**/api/devices/appletv/pair/finish")


def assert_wizard_detection_calibration_and_activation(page) -> None:
    settings_updates = []
    calibrations = []
    starts = []
    setup_updates = []
    profiles = ["espn"]

    def handle_settings(route):
        if route.request.method == "POST":
            settings_updates.append(route.request.post_data_json)
            route.fulfill(status=200, content_type="application/json", body='{"ok":true}')
        else:
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"DETECTION_MODE":"legacy","CALIBRATION_DURATION":60}',
            )

    def handle_profiles(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"profiles": profiles}),
        )

    def handle_calibration(route):
        body = route.request.post_data_json
        calibrations.append(body)
        if body["channel"] not in profiles:
            profiles.append(body["channel"])
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"ok": True, "channel": body["channel"]}),
        )

    def handle_start(route):
        starts.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body='{"ok":true}')

    def handle_setup(route):
        setup_updates.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body='{"ok":true}')

    page.route("**/api/settings", handle_settings)
    page.route("**/api/profiles", handle_profiles)
    page.route(
        "**/api/profile/*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"mode":"logo","corner":"top_right"}',
        ),
    )
    page.route("**/api/calibrate", handle_calibration)
    page.route("**/api/start", handle_start)
    page.route(
        "**/api/stop",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"ok":true}'
        ),
    )
    page.route("**/api/setup", handle_setup)

    page.evaluate(
        """() => {
            wizardStep = 2;
            renderWizard();
            document.querySelector('#setup-wizard').classList.remove('hidden');
        }"""
    )
    page.locator("#wizard-mode").select_option("hybrid")
    page.locator("#wizard-mode-status.success").wait_for()

    page.evaluate("() => { wizardStep = 3; renderWizard(); }")
    page.wait_for_function(
        """() => {
            const select = document.querySelector('#wizard-profile-select');
            return select && !select.disabled && select.options.length === 2;
        }"""
    )
    page.locator("#wizard-profile-select").select_option("espn")
    page.locator("#wizard-calibration-status.success").wait_for()
    page.locator("#wizard-calibration-channel").fill("cnn")
    page.locator("#wizard-calibrate-start").click()
    page.wait_for_function(
        "() => document.querySelector('#wizard-profile-select').value === 'cnn'"
    )

    page.evaluate("() => { wizardStep = 4; renderWizard(); }")
    page.locator("#wizard-shadow-start").click()
    page.locator("#wizard-shadow-status.success").wait_for()

    page.evaluate("() => { wizardStep = 5; renderWizard(); }")
    page.locator("#wizard-next").click()
    page.locator("#setup-wizard").wait_for(state="hidden")

    assert calibrations == [{"channel": "cnn"}]
    assert starts == [
        {"channel": "cnn", "shadow": True, "detection_mode": "hybrid"},
        {"channel": "cnn", "shadow": False, "detection_mode": "hybrid"},
    ]
    assert {"DETECTION_MODE": "hybrid"} in settings_updates
    assert settings_updates[-1] == {
        "DETECTION_MODE": "hybrid",
        "AUTOMATIC_MUTING_ENABLED": True,
        "AUTOMATIC_SKIPPING_ENABLED": False,
    }
    assert setup_updates[-1] == {"complete": True, "step": 6}

    for pattern in (
        "**/api/settings",
        "**/api/profiles",
        "**/api/profile/*",
        "**/api/calibrate",
        "**/api/start",
        "**/api/stop",
        "**/api/setup",
    ):
        page.unroute(pattern)


def run_engine(playwright, name: str):
    if name == "chrome":
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(CHROME),
            args=["--no-sandbox"],
        )
    else:
        browser = playwright.webkit.launch(headless=True)
    try:
        for width in WIDTHS:
            context = browser.new_context(
                viewport={"width": width, "height": 844 if width <= 431 else 900},
                is_mobile=width <= 431,
                has_touch=width <= 768,
                device_scale_factor=2 if width <= 431 else 1,
            )
            page = context.new_page()
            page_errors = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(BASE_URL, wait_until="domcontentloaded")
            page.wait_for_selector("#engine-badge")
            page.wait_for_timeout(500)
            assert page.locator("#setup-wizard").is_hidden()
            assert_layout(page, width)

            if width <= 768:
                page.locator("#bottom-btn-activity").tap()
                assert page.locator("#tab-activity").is_visible()
                page.locator("#bottom-btn-tune").tap()
                assert page.locator("#tab-tune").is_visible()
                assert page.locator("#settings-basic").is_visible()
                page.wait_for_function("() => settingsSnapshot !== null")
                original = page.locator("#basic-auto-muting").is_checked()
                original_skip = page.locator("#basic-auto-skipping").is_checked()
                page.locator("#basic-auto-muting").set_checked(not original)
                page.locator("#basic-auto-skipping").set_checked(not original_skip)
                page.locator("#settings-basic .settings-actions button").first.click()
                page.wait_for_function(
                    "expected => document.querySelector('#basic-auto-muting').checked === expected.mute && document.querySelector('#basic-auto-skipping').checked === expected.skip",
                    arg={"mute": original, "skip": original_skip},
                )
                assert page.locator("#basic-auto-muting").is_checked() == original
                assert page.locator("#basic-auto-skipping").is_checked() == original_skip

            if width == 390:
                assert_wizard_capture_feedback(page)
                assert_wizard_appletv_pairing(page)
                assert_wizard_detection_calibration_and_activation(page)
                page.locator("#bottom-btn-watch").tap()
                page.route("**/api/status", lambda route: route.abort())
                page.wait_for_timeout(2300)
                assert page.locator("#connection-status").is_visible()
                page.unroute("**/api/status")
                page.wait_for_timeout(1300)
                assert page.locator("#connection-status").is_hidden()

            if width in {390, 1024}:
                assert_help_page(context, width)

            if width in {390, 1024}:
                SCREENSHOTS.mkdir(parents=True, exist_ok=True)
                page.screenshot(
                    path=str(SCREENSHOTS / f"{name}-{width}.png"), full_page=True
                )
            assert not page_errors, page_errors
            context.close()
    finally:
        browser.close()


def main() -> None:
    with sync_playwright() as playwright:
        request = playwright.request.new_context(base_url=BASE_URL)
        initial_setup = request.get("/api/setup").json()
        try:
            response = request.post("/api/setup", data={"complete": True, "step": 6})
            assert response.ok
            manifest = request.get("/manifest.webmanifest")
            assert manifest.ok and manifest.json()["display"] == "standalone"
            qr = request.get("/api/network/qr")
            assert qr.ok and qr.headers["content-type"].startswith("image/png")
            run_engine(playwright, "chrome")
            run_engine(playwright, "webkit")
        finally:
            request.post("/api/setup", data=initial_setup)
            request.dispose()
    print("Chrome + WebKit responsive smoke checks passed at 320/390/431/768/1024px")


if __name__ == "__main__":
    main()
