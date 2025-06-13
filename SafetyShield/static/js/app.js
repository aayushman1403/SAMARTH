/**
 * Mumbai Safety Zone Predictor - Application Controller
 * 
 * This module manages the overall application, coordinating between
 * the map, time controls, and other UI components.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the map using the static method which exposes it globally
    const safetyMap = SafetyMap.initialize('map');
    
    // Set up time controls
    setupTimeControls(safetyMap);
    
    // Set up other UI components
    setupUIComponents();
});

/**
 * Set up time control slider and buttons
 */
function setupTimeControls(safetyMap) {
    const timeSlider = document.getElementById('time-slider');
    const hourDisplay = document.getElementById('current-hour-display');
    const playButton = document.getElementById('play-button');
    const pauseButton = document.getElementById('pause-button');
    
    let playInterval = null;
    let isPlaying = false;
    
    // Initialize slider to current hour
    const currentHour = new Date().getHours();
    timeSlider.value = currentHour;
    updateHourDisplay(currentHour, hourDisplay);
    
    // Update map when slider changes
    timeSlider.addEventListener('input', function() {
        const hour = parseInt(this.value);
        updateHourDisplay(hour, hourDisplay);
        safetyMap.updateSafetyLevels(hour);
    });
    
    // Play button - animate through 24 hours
    playButton.addEventListener('click', function() {
        if (isPlaying) return;
        
        isPlaying = true;
        playButton.disabled = true;
        pauseButton.disabled = false;
        
        // Animate through all hours
        let hour = parseInt(timeSlider.value);
        
        playInterval = setInterval(() => {
            hour = (hour + 1) % 24;
            timeSlider.value = hour;
            updateHourDisplay(hour, hourDisplay);
            safetyMap.updateSafetyLevels(hour);
        }, 1500); // 1.5 seconds per hour
    });
    
    // Pause button - stop animation
    pauseButton.addEventListener('click', function() {
        clearInterval(playInterval);
        isPlaying = false;
        playButton.disabled = false;
        pauseButton.disabled = true;
    });
    
    // Time preset buttons
    document.querySelectorAll('.time-preset').forEach(button => {
        button.addEventListener('click', function() {
            // Stop any ongoing animation
            if (isPlaying) {
                pauseButton.click();
            }
            
            const hour = parseInt(this.dataset.hour);
            timeSlider.value = hour;
            updateHourDisplay(hour, hourDisplay);
            safetyMap.updateSafetyLevels(hour);
        });
    });
}

/**
 * Update the hour display with formatted time
 */
function updateHourDisplay(hour, displayElement) {
    // Format hour as "HH:00"
    const formattedHour = hour.toString().padStart(2, '0');
    displayElement.textContent = `${formattedHour}:00 Hours`;
}

/**
 * Set up other UI components and interactions
 */
function setupUIComponents() {
    // Theme toggle for light/dark mode
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const body = document.body;
            if (body.getAttribute('data-bs-theme') === 'dark') {
                body.setAttribute('data-bs-theme', 'light');
                themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            } else {
                body.setAttribute('data-bs-theme', 'dark');
                themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            }
        });
    }
    
    // Information panel toggle
    const infoToggle = document.getElementById('info-panel-toggle');
    const infoPanel = document.getElementById('info-panel');
    
    if (infoToggle && infoPanel) {
        infoToggle.addEventListener('click', function() {
            infoPanel.classList.toggle('show');
            
            // Update toggle button icon and text
            if (infoPanel.classList.contains('show')) {
                infoToggle.innerHTML = '<i class="fas fa-times"></i> Close';
            } else {
                infoToggle.innerHTML = '<i class="fas fa-info-circle"></i> Info';
            }
        });
    }
    
    // Close alerts
    document.querySelectorAll('.alert .close').forEach(button => {
        button.addEventListener('click', function() {
            this.parentElement.style.display = 'none';
        });
    });
}
