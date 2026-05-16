// Fix for AudioContext sample rate mismatch
// This ensures all AudioContext instances use the same sample rate

(function() {
    console.log('[Audio Fix] Initializing AudioContext fix...');
    
    // Store the original AudioContext constructor
    const OriginalAudioContext = window.AudioContext || window.webkitAudioContext;
    
    if (!OriginalAudioContext) {
        console.warn('[Audio Fix] AudioContext not supported');
        return;
    }
    
    // Create a single shared AudioContext with the browser's default sample rate
    let sharedContext = null;
    
    function getSharedContext() {
        if (!sharedContext || sharedContext.state === 'closed') {
            sharedContext = new OriginalAudioContext();
            console.log('[Audio Fix] Created shared AudioContext with sample rate:', sharedContext.sampleRate);
        }
        return sharedContext;
    }
    
    // Override the AudioContext constructor
    const CustomAudioContext = function(options) {
        console.log('[Audio Fix] AudioContext constructor called with options:', options);
        
        // Always use the shared context to avoid sample rate mismatches
        const ctx = getSharedContext();
        
        // If options specified a different sample rate, log a warning
        if (options && options.sampleRate && options.sampleRate !== ctx.sampleRate) {
            console.warn(
                '[Audio Fix] Requested sample rate', options.sampleRate,
                'differs from browser default', ctx.sampleRate,
                '- using browser default to avoid errors'
            );
        }
        
        return ctx;
    };
    
    // Copy prototype and static properties
    CustomAudioContext.prototype = OriginalAudioContext.prototype;
    Object.setPrototypeOf(CustomAudioContext, OriginalAudioContext);
    
    // Replace the global AudioContext
    window.AudioContext = CustomAudioContext;
    if (window.webkitAudioContext) {
        window.webkitAudioContext = CustomAudioContext;
    }
    
    console.log('[Audio Fix] AudioContext override complete');
})();
