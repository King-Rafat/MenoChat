/* ==========================================================================
   VOICE-ONLY MODE: seamless conversation loop
   --------------------------------------------------------------------------
   Kid version: when the bot starts speaking, we close the user's mic so we
   don't record the bot's own voice. When the bot finishes speaking, we
   re-open the mic so the user can answer back. The user only ever has to
   tap the big button ONCE to start the whole conversation.
   ========================================================================== */

(function () {
    const MIC_SELECTORS = [
        'button[aria-label*="Microphone" i]',
        'button[aria-label*="microphone" i]',
        'button[aria-label*="record" i]',
        'button[aria-label*="Stop" i]',  // when actively recording, label flips to Stop
        '[data-testid="record-button"]',
        '[data-testid="mic-button"]',
    ];

    function findMic() {
        for (const s of MIC_SELECTORS) {
            const el = document.querySelector(s);
            if (el) return el;
        }
        return null;
    }

    // We track our own belief about whether the mic is on, because Chainlit
    // doesn't expose a stable attribute for it across versions. Every time
    // someone clicks the mic button (the user OR our auto-click), we flip
    // this boolean.
    let micOn = false;

    document.addEventListener('click', (e) => {
        if (!(e.target instanceof Element)) return;
        const target = e.target.closest(MIC_SELECTORS.join(','));
        if (target) {
            micOn = !micOn;
            // Add a class so CSS pulse animation can hook in if aria-label
            // doesn't change between Microphone/Stop on this Chainlit build.
            target.classList.toggle('is-recording', micOn);
            console.log('[voice-only] mic toggled ->', micOn);
        }
    }, true);

    function clickMic() {
        const btn = findMic();
        if (btn) btn.click();
    }

    // Hook every <audio> element Chainlit drops into the DOM (TTS responses).
    const seen = new WeakSet();
    function hookAudio(audio) {
        if (seen.has(audio)) return;
        seen.add(audio);

        audio.addEventListener('play', () => {
            // Bot started speaking. Close the mic so we don't record the bot.
            if (micOn) {
                console.log('[voice-only] TTS started, closing mic');
                clickMic();
            }
        });

        audio.addEventListener('ended', () => {
            // Bot finished speaking. Re-open the mic for the next turn.
            setTimeout(() => {
                if (!micOn) {
                    console.log('[voice-only] TTS ended, reopening mic');
                    clickMic();
                }
            }, 400);
        });

        audio.addEventListener('error', () => {
            setTimeout(() => { if (!micOn) clickMic(); }, 400);
        });
    }

    new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (!(node instanceof HTMLElement)) continue;
                if (node.tagName === 'AUDIO') {
                    hookAudio(node);
                } else if (node.querySelectorAll) {
                    node.querySelectorAll('audio').forEach(hookAudio);
                }
            }
        }
    }).observe(document.body, { childList: true, subtree: true });

    console.log('[voice-only] seamless loop armed');
})();
