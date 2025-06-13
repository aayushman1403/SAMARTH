/**
 * Mumbai Safety Zone Predictor - Map Module
 * 
 * This module handles the map initialization, rendering, and updating of
 * safety zones based on crime prediction data.
 */

class SafetyMap {
    constructor(mapId) {
        this.mapId = mapId;
        this.map = null;
        this.wardLayers = {};
        this.wardMarkers = {};
        this.legendControl = null;
        this.searchControl = null;
        this.selectedWard = null;
        this.safetyColors = {
            green: '#28a745',   // Safe
            yellow: '#ffc107',  // Moderate risk
            red: '#dc3545'      // High risk
        };
        this.defaultStyle = {
            weight: 1,
            opacity: 0.8,
            color: '#666',
            fillOpacity: 0.4
        };
        this.highlightStyle = {
            weight: 3,
            color: '#fff',
            fillOpacity: 0.7
        };
        this.markerOptions = {
            radius: 10,        // 10m radius (very small and clear)
            stroke: true,
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        };
        this.currentHour = new Date().getHours();
    }

    /**
     * Initialize the Leaflet map
     */
    init() {
        // Create map centered on Mumbai
        this.map = L.map(this.mapId, {
            center: [19.0760, 72.8777], // Mumbai coordinates
            zoom: 13,
            minZoom: 10,
            maxZoom: 16
        });

        // Add basemap layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);

        // Add search control
        this.setupSearchControl();

        // Add legend
        this.addLegend();

        // Load initial ward data
        this.loadWardData();
    }

    /**
     * Set up the search control
     */
    setupSearchControl() {
        // Create custom search box
        const searchControlHTML = `
            <div class="leaflet-control-search">
                <input type="text" id="location-search" class="form-control" placeholder="Search for a location...">
                <button id="search-btn" class="btn btn-primary btn-sm"><i class="fas fa-search"></i></button>
            </div>
        `;

        // Create custom control
        const SearchControl = L.Control.extend({
            options: {
                position: 'topleft'
            },
            onAdd: (map) => {
                const container = L.DomUtil.create('div', 'leaflet-control leaflet-control-search-container');
                container.innerHTML = searchControlHTML;
                
                // Prevent map dragging when interacting with the search control
                L.DomEvent.disableClickPropagation(container);
                
                return container;
            }
        });
        
        // Add the control to the map
        this.searchControl = new SearchControl();
        this.map.addControl(this.searchControl);
        
        // Set up event listener for search button
        setTimeout(() => {
            const searchBtn = document.getElementById('search-btn');
            const searchInput = document.getElementById('location-search');
            
            if (searchBtn && searchInput) {
                searchBtn.addEventListener('click', () => {
                    this.searchLocation(searchInput.value);
                });
                
                searchInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.searchLocation(searchInput.value);
                    }
                });
            }
        }, 100);
    }

    /**
     * Search for a location and highlight the corresponding ward
     */
    searchLocation(query) {
        if (!query || query.trim() === '') {
            this.showErrorOverlay('Please enter a location to search');
            return;
        }
        
        // Show loading indicator
        const searchInput = document.getElementById('location-search');
        if (searchInput) {
            searchInput.classList.add('loading');
        }
        
        // Call the backend API to search for the location
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Location search failed');
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Remove loading state
                if (searchInput) {
                    searchInput.classList.remove('loading');
                }
                
                // Highlight the ward and zoom to it
                this.highlightSearchResult(data);
            })
            .catch(error => {
                console.error('Error searching location:', error);
                if (searchInput) {
                    searchInput.classList.remove('loading');
                }
                this.showErrorOverlay(`Search failed: ${error.message}`);
            });
    }

    /**
     * Highlight a ward based on search result
     */
    highlightSearchResult(searchResult) {
        const { ward_id, name, search_lat, search_lng, matched_location } = searchResult;
        
        // Clear any previous search marker
        if (this.searchMarker) {
            this.map.removeLayer(this.searchMarker);
        }
        
        // Create a marker at the search location
        this.searchMarker = L.marker([search_lat, search_lng], {
            icon: L.divIcon({
                className: 'search-result-marker',
                html: '<i class="fas fa-search-location"></i>',
                iconSize: [30, 30],
                iconAnchor: [15, 30]
            })
        }).addTo(this.map);
        
        // Add popup to the marker
        this.searchMarker.bindPopup(`
            <div class="search-result-popup">
                <h6>${matched_location}</h6>
                <p>Located in <strong>${name}</strong></p>
                <div class="mt-2">
                    <button class="btn btn-sm btn-primary" id="show-all-wards-btn">Show All Regions</button>
                </div>
            </div>
        `).openPopup();
        
        // Zoom to the location
        this.map.setView([search_lat, search_lng], 14);
        
        // Hide all other ward markers except the selected one
        Object.keys(this.wardMarkers).forEach(key => {
            if (key !== ward_id) {
                this.wardMarkers[key].setStyle({opacity: 0, fillOpacity: 0});
            } else {
                // Make the selected ward more prominent
                this.wardMarkers[key].setStyle({
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 0.9
                });
            }
        });
        
        // Highlight the corresponding ward
        if (ward_id in this.wardMarkers) {
            // Simulate a click on the ward marker
            this.onWardClick({ target: this.wardMarkers[ward_id] });
        }
        
        // Set up event listener for "Show All Regions" button
        setTimeout(() => {
            const showAllBtn = document.getElementById('show-all-wards-btn');
            if (showAllBtn) {
                showAllBtn.addEventListener('click', () => {
                    this.showAllWards();
                });
            }
        }, 100);
        
        // Store the currently visible ward
        this.visibleWardId = ward_id;
        
        // Show info toast
        this.showInfoToast(`Location found in ${name}. Other regions hidden.`);
    }
    
    /**
     * Show all ward markers (restore visibility)
     */
    showAllWards() {
        Object.keys(this.wardMarkers).forEach(key => {
            const marker = this.wardMarkers[key];
            const safetyLevel = marker.safetyLevel || 'green';
            
            marker.setStyle({
                weight: this.markerOptions.weight,
                color: this.defaultStyle.color,
                opacity: 1,
                fillOpacity: 0.8,
                fillColor: this.safetyColors[safetyLevel]
            });
        });
        
        this.visibleWardId = null;
        
        // Zoom out to show all wards
        if (this.wardBoundaries) {
            this.map.fitBounds(this.wardBoundaries.getBounds());
        }
        
        // Show info toast
        this.showInfoToast("All regions are now visible.");
    }

    /**
     * Show info toast message
     */
    showInfoToast(message) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            const container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(container);
        }
        
        const toastId = `toast-${Date.now()}`;
        const toastHTML = `
            <div class="toast" id="${toastId}" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header">
                    <i class="fas fa-info-circle text-info me-2"></i>
                    <strong class="me-auto">Mumbai Safety Predictor</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        document.getElementById('toast-container').innerHTML += toastHTML;
        
        // Initialize and show the toast
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
        toast.show();
        
        // Remove toast after it's hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    /**
     * Load ward boundaries GeoJSON and set up initial styling
     */
    loadWardData() {
        fetch('/api/wards')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ward data not available');
                }
                return response.json();
            })
            .then(data => {
                this.setupWardLayers(data);
                this.setupWardMarkers(data);
                this.updateSafetyLevels(this.currentHour);
            })
            .catch(error => {
                console.error('Error loading ward data:', error);
                // Display error message on the map
                this.showErrorOverlay('Unable to load Mumbai ward data. Please try again later.');
            });
    }

    /**
     * Set up the GeoJSON layers for ward boundaries
     */
    setupWardLayers(geojson) {
        // Create a semi-transparent layer for the ward boundaries
        const wardLayer = L.geoJSON(geojson, {
            style: {
                ...this.defaultStyle,
                fillOpacity: 0.2,  // Very transparent
                fillColor: '#888' 
            }
        }).addTo(this.map);

        // Fit map to ward boundaries
        this.map.fitBounds(wardLayer.getBounds());
        
        // Store ward boundaries in a separate variable (not shown by default)
        this.wardBoundaries = wardLayer;
    }
    
    /**
     * Set up circular markers for each ward
     */
    setupWardMarkers(geojson) {
        // Clear any existing markers
        if (this.markerGroup) {
            this.map.removeLayer(this.markerGroup);
        }
        
        // Create a layer group for markers
        this.markerGroup = L.layerGroup().addTo(this.map);
        
        // Create a marker for each ward
        geojson.features.forEach(feature => {
            const wardId = feature.properties.ward_id;
            const name = feature.properties.name || `Ward ${wardId}`;
            
            // Get center of ward polygon (calculate centroid)
            let centerLat = 19.0760; // Default to Mumbai center
            let centerLng = 72.8777;
            
            try {
                const coords = feature.geometry.coordinates[0];
                if (coords && coords.length > 0) {
                    const lats = coords.map(coord => coord[1]);
                    const lngs = coords.map(coord => coord[0]);
                    centerLat = lats.reduce((a, b) => a + b, 0) / lats.length;
                    centerLng = lngs.reduce((a, b) => a + b, 0) / lngs.length;
                }
            } catch (e) {
                console.warn(`Error calculating centroid for ward ${wardId}`, e);
            }
            
            // Create circle marker
            const marker = L.circleMarker([centerLat, centerLng], {
                ...this.markerOptions,
                fillColor: this.safetyColors.green // Default, will be updated
            }).addTo(this.markerGroup);
            
            // Store ward ID and properties in the marker
            marker.wardId = wardId;
            marker.wardName = name;
            marker.safetyLevel = 'green'; // Default, will be updated
            
            // Add popup
            marker.bindPopup(this.createWardPopup({
                ward_id: wardId,
                name: name
            }));
            
            // Add event listeners
            marker.on({
                mouseover: (e) => this.highlightWard(e),
                mouseout: (e) => this.resetWardHighlight(e),
                click: (e) => this.onWardClick(e)
            });
            
            // Store reference to this marker
            this.wardMarkers[wardId] = marker;
        });
    }

    /**
     * Create popup content for ward
     */
    createWardPopup(properties) {
        return `
            <div class="ward-popup">
                <h5>${properties.name || 'Ward ' + properties.ward_id}</h5>
                <p><strong>ID:</strong> ${properties.ward_id}</p>
                <div class="safety-info">
                    <p><strong>Safety Level:</strong> <span class="safety-level">Green</span></p>
                    <p><strong>Risk Factors:</strong> <span class="risk-factors">None identified</span></p>
                </div>
                <div class="popup-actions mt-2">
                    <button class="btn btn-sm btn-info view-history-btn" data-ward-id="${properties.ward_id}">
                        <i class="fas fa-chart-line"></i> Historical Data
                    </button>
                    <button class="btn btn-sm btn-warning view-tips-btn" data-ward-id="${properties.ward_id}">
                        <i class="fas fa-shield-alt"></i> Safety Tips
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Update ward style on hover
     */
    highlightWard(e) {
        const marker = e.target;
        
        // Make border more prominent without increasing size
        marker.setStyle({
            weight: 3,
            color: '#fff'
        });
        
        // Bring to front
        marker.bringToFront();
        
        // Show ward name tooltip if not already showing popup
        if (!marker._popup || !marker._popup._isOpen) {
            marker.unbindTooltip();
            marker.bindTooltip(marker.wardName, {
                permanent: false,
                direction: 'top',
                className: 'ward-tooltip'
            }).openTooltip();
        }
    }

    /**
     * Reset ward style after hover
     */
    resetWardHighlight(e) {
        const marker = e.target;
        const safetyLevel = marker.safetyLevel || 'green';
        
        // Reset to original style
        marker.setStyle({
            weight: this.markerOptions.weight,
            color: this.defaultStyle.color,
            fillColor: this.safetyColors[safetyLevel]
        });
        
        // Close tooltip
        marker.closeTooltip();
    }

    /**
     * Handle ward click event
     */
    onWardClick(e) {
        const marker = e.target;
        const wardId = marker.wardId;
        const wardName = marker.wardName;
        
        // Store the selected ward
        this.selectedWard = wardId;
        
        // Make sure popup is open
        marker.openPopup();
        
        // Fetch and display detailed ward information in the sidebar
        this.updateWardDetailsSidebar(wardId, wardName);
        
        // Attach event listeners to the popup buttons
        setTimeout(() => {
            const historyBtn = document.querySelector(`.view-history-btn[data-ward-id="${wardId}"]`);
            const tipsBtn = document.querySelector(`.view-tips-btn[data-ward-id="${wardId}"]`);
            
            if (historyBtn) {
                historyBtn.addEventListener('click', () => {
                    this.showHistoricalData(wardId, wardName);
                });
            }
            
            if (tipsBtn) {
                tipsBtn.addEventListener('click', () => {
                    this.showSafetyTips(wardId, wardName);
                });
            }
        }, 100);
    }
    
    /**
     * Update the ward details sidebar with information
     */
    updateWardDetailsSidebar(wardId, wardName) {
        const detailsPanel = document.getElementById('ward-details');
        const wardNameEl = document.getElementById('ward-name');
        
        if (detailsPanel && wardNameEl) {
            // Show the panel
            detailsPanel.classList.remove('d-none');
            
            // Update the ward name
            wardNameEl.textContent = wardName;
            
            // Update safety level based on current hour predictions
            const hour = parseInt(document.getElementById('time-slider').value);
            fetch(`/api/predict?hour=${hour}`)
                .then(response => response.json())
                .then(data => {
                    const wardData = data.wards.find(w => w.ward_id === wardId);
                    if (wardData) {
                        // Update safety level badge
                        const safetyLevelBadge = document.getElementById('ward-safety-level');
                        if (safetyLevelBadge) {
                            safetyLevelBadge.textContent = wardData.safety_level.toUpperCase();
                            safetyLevelBadge.className = `badge rounded-pill bg-${wardData.safety_level}`;
                        }
                        
                        // Update crime probability
                        const probabilityEl = document.getElementById('ward-crime-probability');
                        if (probabilityEl) {
                            probabilityEl.textContent = `${Math.round(wardData.crime_probability * 100)}%`;
                        }
                        
                        // Update risk factors
                        const riskFactorsEl = document.getElementById('ward-risk-factors');
                        if (riskFactorsEl && wardData.risk_factors) {
                            let factorsHtml = '';
                            wardData.risk_factors.forEach(factor => {
                                factorsHtml += `<li><i class="fas fa-exclamation-triangle text-warning me-2"></i> ${factor}</li>`;
                            });
                            riskFactorsEl.innerHTML = factorsHtml;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching ward details:', error);
                });
                
            // Fetch future predictions
            fetch(`/api/future/${wardId}?hours=6`)
                .then(response => response.json())
                .then(data => {
                    if (data.predictions) {
                        // Display future predictions
                        const futureContainer = document.getElementById('future-predictions');
                        if (futureContainer) {
                            let predictionsHtml = '<h6>Upcoming Hours Forecast</h6><div class="d-flex justify-content-between">';
                            
                            // Show the next few hours
                            data.predictions.slice(0, 6).forEach(prediction => {
                                const hour = new Date(prediction.timestamp).getHours();
                                predictionsHtml += `
                                    <div class="future-hour text-center mx-1">
                                        <div class="hour-label">${hour}:00</div>
                                        <div class="safety-indicator" style="background-color: ${this.safetyColors[prediction.safety_level]}"></div>
                                    </div>
                                `;
                            });
                            
                            predictionsHtml += '</div>';
                            futureContainer.innerHTML = predictionsHtml;
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching future predictions:', error);
                });
        }
    }
    
    /**
     * Show historical data for the ward
     */
    showHistoricalData(wardId, wardName) {
        // Create a modal to display historical data
        const modalId = 'historical-data-modal';
        let modalHTML = `
            <div class="modal fade" id="${modalId}" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Historical Safety Data: ${wardName}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div class="text-center mb-3">
                                <div class="btn-group" role="group">
                                    <button type="button" class="btn btn-outline-primary history-days-btn active" data-days="7">Last 7 Days</button>
                                    <button type="button" class="btn btn-outline-primary history-days-btn" data-days="14">Last 14 Days</button>
                                    <button type="button" class="btn btn-outline-primary history-days-btn" data-days="30">Last 30 Days</button>
                                </div>
                            </div>
                            <div id="historical-data-content">
                                <div class="text-center py-4">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    <p class="mt-2">Loading historical data...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document if it doesn't exist
        if (!document.getElementById(modalId)) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }
        
        // Initialize and show the modal
        const modal = new bootstrap.Modal(document.getElementById(modalId));
        modal.show();
        
        // Load historical data (default to 7 days)
        this.loadHistoricalData(wardId, wardName, 7);
        
        // Add event listeners to day selection buttons
        setTimeout(() => {
            document.querySelectorAll('.history-days-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    // Update active button
                    document.querySelectorAll('.history-days-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    
                    // Load data for selected days
                    const days = parseInt(e.target.dataset.days);
                    this.loadHistoricalData(wardId, wardName, days);
                });
            });
        }, 100);
    }
    
    /**
     * Load historical data for the ward
     */
    loadHistoricalData(wardId, wardName, days) {
        const contentEl = document.getElementById('historical-data-content');
        if (contentEl) {
            // Show loading state
            contentEl.innerHTML = `
                <div class="text-center py-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading ${days} days of historical data...</p>
                </div>
            `;
        }
        
        // Fetch historical data
        fetch(`/api/historical/${wardId}?days=${days}`)
            .then(response => response.json())
            .then(data => {
                if (contentEl) {
                    if (data.error) {
                        contentEl.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        return;
                    }
                    
                    // Display period stats
                    let periodStatsHtml = `
                        <h5 class="mb-3">Safety Overview (${days} Days)</h5>
                        <div class="row mb-4">
                    `;
                    
                    // Display stats for different time periods
                    for (const [period, stats] of Object.entries(data.period_stats)) {
                        const periodTitle = period.charAt(0).toUpperCase() + period.slice(1);
                        const dominantClass = stats.dominant_safety;
                        
                        periodStatsHtml += `
                            <div class="col-md-3 mb-3">
                                <div class="card h-100">
                                    <div class="card-header bg-${dominantClass} text-white">
                                        <h6 class="mb-0">${periodTitle}</h6>
                                    </div>
                                    <div class="card-body">
                                        <p class="mb-1">Dominant: ${stats.dominant_safety.toUpperCase()} (${stats.dominant_percentage}%)</p>
                                        <div class="safety-distribution">
                                            <div class="progress mb-2" style="height: 20px;">
                                                <div class="progress-bar bg-success" style="width: ${stats.green_pct}%" title="Safe: ${stats.green_pct}%"></div>
                                                <div class="progress-bar bg-warning" style="width: ${stats.yellow_pct}%" title="Caution: ${stats.yellow_pct}%"></div>
                                                <div class="progress-bar bg-danger" style="width: ${stats.red_pct}%" title="High Risk: ${stats.red_pct}%"></div>
                                            </div>
                                            <div class="d-flex justify-content-between small">
                                                <span>S: ${stats.green_pct}%</span>
                                                <span>C: ${stats.yellow_pct}%</span>
                                                <span>R: ${stats.red_pct}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    periodStatsHtml += `</div>`;
                    
                    // Display daily trend (simplified)
                    let dailyTrendHtml = `<h5 class="mb-3">Daily Safety Trend</h5><div class="daily-trend mb-4">`;
                    
                    // Create a heat map representation of daily data
                    data.daily_data.forEach(day => {
                        const date = new Date(day.date);
                        const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                        
                        dailyTrendHtml += `
                            <div class="day-column">
                                <div class="day-header">${formattedDate}</div>
                                <div class="hours-container">
                        `;
                        
                        // Group hours into 4-hour blocks for simplicity
                        const hourGroups = [
                            {name: "Night", hours: [0, 1, 2, 3]},
                            {name: "Early", hours: [4, 5, 6, 7]},
                            {name: "Morning", hours: [8, 9, 10, 11]},
                            {name: "Afternoon", hours: [12, 13, 14, 15]},
                            {name: "Evening", hours: [16, 17, 18, 19]},
                            {name: "Late", hours: [20, 21, 22, 23]}
                        ];
                        
                        hourGroups.forEach(group => {
                            // Count safety levels in this group
                            let greenCount = 0, yellowCount = 0, redCount = 0;
                            group.hours.forEach(hour => {
                                const hourData = day.hourly_data.find(h => h.hour === hour);
                                if (hourData) {
                                    if (hourData.safety_level === 'green') greenCount++;
                                    else if (hourData.safety_level === 'yellow') yellowCount++;
                                    else redCount++;
                                }
                            });
                            
                            // Determine dominant safety level
                            let dominantLevel = 'green';
                            if (yellowCount >= greenCount && yellowCount >= redCount) {
                                dominantLevel = 'yellow';
                            } else if (redCount >= greenCount && redCount >= yellowCount) {
                                dominantLevel = 'red';
                            }
                            
                            dailyTrendHtml += `
                                <div class="hour-block bg-${dominantLevel}" title="${group.name}: ${group.hours[0]}:00-${group.hours[3]}:00">
                                    <span class="hour-label">${group.name}</span>
                                </div>
                            `;
                        });
                        
                        dailyTrendHtml += `</div></div>`;
                    });
                    
                    dailyTrendHtml += `</div>`;
                    
                    // Combine all sections
                    contentEl.innerHTML = periodStatsHtml + dailyTrendHtml;
                    
                    // Add some basic styles if not already added
                    if (!document.getElementById('historical-data-styles')) {
                        const stylesEl = document.createElement('style');
                        stylesEl.id = 'historical-data-styles';
                        stylesEl.textContent = `
                            .daily-trend {
                                display: flex;
                                overflow-x: auto;
                                padding-bottom: 10px;
                            }
                            .day-column {
                                min-width: 80px;
                                margin-right: 5px;
                            }
                            .day-header {
                                text-align: center;
                                font-weight: bold;
                                margin-bottom: 5px;
                            }
                            .hours-container {
                                display: flex;
                                flex-direction: column;
                            }
                            .hour-block {
                                height: 30px;
                                margin-bottom: 3px;
                                border-radius: 3px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 0.8rem;
                            }
                            .bg-green { background-color: #28a745; }
                            .bg-yellow { background-color: #ffc107; color: #212529; }
                            .bg-red { background-color: #dc3545; }
                        `;
                        document.head.appendChild(stylesEl);
                    }
                }
            })
            .catch(error => {
                if (contentEl) {
                    contentEl.innerHTML = `<div class="alert alert-danger">Failed to load historical data: ${error.message}</div>`;
                }
            });
    }
    
    /**
     * Show safety tips for the ward
     */
    showSafetyTips(wardId, wardName) {
        // Create a modal to display safety tips
        const modalId = 'safety-tips-modal';
        let modalHTML = `
            <div class="modal fade" id="${modalId}" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Safety Tips: ${wardName}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="safety-tips-content">
                                <div class="text-center py-4">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    <p class="mt-2">Loading safety tips...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document if it doesn't exist
        if (!document.getElementById(modalId)) {
            document.body.insertAdjacentHTML('beforeend', modalHTML);
        }
        
        // Initialize and show the modal
        const modal = new bootstrap.Modal(document.getElementById(modalId));
        modal.show();
        
        // Get current hour from slider
        const hour = parseInt(document.getElementById('time-slider').value);
        
        // Fetch safety tips
        fetch(`/api/tips/${wardId}?hour=${hour}`)
            .then(response => response.json())
            .then(data => {
                const contentEl = document.getElementById('safety-tips-content');
                if (contentEl) {
                    if (data.error) {
                        contentEl.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        return;
                    }
                    
                    // Build safety tips content
                    let tipsHtml = `
                        <div class="safety-status mb-3">
                            <div class="alert alert-${data.safety_level}" role="alert">
                                <h5 class="mb-1">Current Status: ${data.safety_level.toUpperCase()}</h5>
                                <p class="mb-0">For ${data.ward_name} at ${hour}:00 hours</p>
                            </div>
                        </div>
                    `;
                    
                    // General tips
                    tipsHtml += `
                        <div class="general-tips mb-3">
                            <h6>General Safety Tips</h6>
                            <ul class="list-group">
                    `;
                    
                    data.general_tips.forEach(tip => {
                        tipsHtml += `<li class="list-group-item"><i class="fas fa-check-circle text-success me-2"></i> ${tip}</li>`;
                    });
                    
                    tipsHtml += `</ul></div>`;
                    
                    // Specific tips for this location and time
                    if (data.specific_tips && data.specific_tips.length > 0) {
                        tipsHtml += `
                            <div class="specific-tips mb-3">
                                <h6>Location-Specific Tips</h6>
                                <ul class="list-group">
                        `;
                        
                        data.specific_tips.forEach(tip => {
                            tipsHtml += `<li class="list-group-item"><i class="fas fa-map-marker-alt text-danger me-2"></i> ${tip}</li>`;
                        });
                        
                        tipsHtml += `</ul></div>`;
                    }
                    
                    // Time-specific tips
                    if (data.time_tips && data.time_tips.length > 0) {
                        tipsHtml += `
                            <div class="time-tips mb-3">
                                <h6>Time-Specific Tips (${hour}:00)</h6>
                                <ul class="list-group">
                        `;
                        
                        data.time_tips.forEach(tip => {
                            tipsHtml += `<li class="list-group-item"><i class="fas fa-clock text-warning me-2"></i> ${tip}</li>`;
                        });
                        
                        tipsHtml += `</ul></div>`;
                    }
                    
                    // Emergency contacts section
                    tipsHtml += `
                        <div class="emergency-contacts mt-4">
                            <h6>Emergency Contacts</h6>
                            <div class="list-group">
                                <a href="tel:100" class="list-group-item list-group-item-action">
                                    <i class="fas fa-phone-alt text-danger me-2"></i> Police: 100
                                </a>
                                <a href="tel:108" class="list-group-item list-group-item-action">
                                    <i class="fas fa-ambulance text-danger me-2"></i> Ambulance: 108
                                </a>
                                <a href="tel:101" class="list-group-item list-group-item-action">
                                    <i class="fas fa-fire text-danger me-2"></i> Fire: 101
                                </a>
                                <a href="tel:1091" class="list-group-item list-group-item-action">
                                    <i class="fas fa-venus text-danger me-2"></i> Women's Helpline: 1091
                                </a>
                            </div>
                        </div>
                    `;
                    
                    contentEl.innerHTML = tipsHtml;
                }
            })
            .catch(error => {
                const contentEl = document.getElementById('safety-tips-content');
                if (contentEl) {
                    contentEl.innerHTML = `<div class="alert alert-danger">Failed to load safety tips: ${error.message}</div>`;
                }
            });
    }

    /**
     * Update the safety colors of wards based on prediction for the given hour
     */
    updateSafetyLevels(hour) {
        fetch(`/api/predict?hour=${hour}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Prediction data not available');
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Update the map UI to show current time
                document.getElementById('current-hour-display').textContent = 
                    `${data.timestamp} Hours`;
                
                // Update each ward's safety level on the map
                data.wards.forEach(ward => {
                    const marker = this.wardMarkers[ward.ward_id];
                    if (marker) {
                        // Store safety level on the marker for reference
                        marker.safetyLevel = ward.safety_level;
                        
                        // Update marker style
                        marker.setStyle({
                            fillColor: this.safetyColors[ward.safety_level]
                        });
                        
                        // Update popup content if it's open
                        if (marker._popup && marker._popup._isOpen) {
                            this.updatePopupContent(marker._popup, ward);
                        }
                    }
                });
                
                // Update the legend
                this.updateLegendCounts(data.wards);
                
                // If there's a selected ward, update the sidebar too
                if (this.selectedWard && this.selectedWard in this.wardMarkers) {
                    const wardId = this.selectedWard;
                    const wardName = this.wardMarkers[wardId].wardName;
                    this.updateWardDetailsSidebar(wardId, wardName);
                }
            })
            .catch(error => {
                console.error('Error updating safety levels:', error);
                // Show error message
                this.showErrorOverlay(`Unable to load prediction data: ${error.message}`);
            });
    }

    /**
     * Update popup content with safety information
     */
    updatePopupContent(popup, wardData) {
        const popupContent = popup.getContent();
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = popupContent;
        
        // Update safety level with colored indicator
        const safetyLevelEl = tempDiv.querySelector('.safety-level');
        if (safetyLevelEl) {
            safetyLevelEl.textContent = wardData.safety_level.toUpperCase();
            safetyLevelEl.style.color = this.safetyColors[wardData.safety_level];
        }
        
        // Update risk factors
        const riskFactorsEl = tempDiv.querySelector('.risk-factors');
        if (riskFactorsEl) {
            riskFactorsEl.textContent = wardData.risk_factors 
                ? wardData.risk_factors.join(', ') 
                : 'None identified';
        }
        
        popup.setContent(tempDiv.innerHTML);
    }

    /**
     * Add legend to the map
     */
    addLegend() {
        const legendControl = L.control({position: 'bottomright'});
        
        legendControl.onAdd = (map) => {
            const div = L.DomUtil.create('div', 'info legend');
            div.innerHTML = `
                <div class="card">
                    <div class="card-body p-2">
                        <h6 class="card-title">Safety Levels</h6>
                        <div class="safety-level-item">
                            <i style="background:${this.safetyColors.green}"></i>
                            Safe <span class="green-count badge bg-secondary">0</span>
                        </div>
                        <div class="safety-level-item">
                            <i style="background:${this.safetyColors.yellow}"></i>
                            Caution <span class="yellow-count badge bg-secondary">0</span>
                        </div>
                        <div class="safety-level-item">
                            <i style="background:${this.safetyColors.red}"></i>
                            High Risk <span class="red-count badge bg-secondary">0</span>
                        </div>
                    </div>
                </div>
            `;
            return div;
        };
        
        legendControl.addTo(this.map);
        this.legendControl = legendControl;
    }

    /**
     * Update counts in the legend
     */
    updateLegendCounts(wards) {
        if (!this.legendControl) return;
        
        // Count wards by safety level
        const counts = {
            green: 0,
            yellow: 0,
            red: 0
        };
        
        wards.forEach(ward => {
            counts[ward.safety_level] += 1;
        });
        
        // Update the count badges
        const legend = document.querySelector('.legend');
        if (legend) {
            legend.querySelector('.green-count').textContent = counts.green;
            legend.querySelector('.yellow-count').textContent = counts.yellow;
            legend.querySelector('.red-count').textContent = counts.red;
        }
    }

    /**
     * Show error overlay on map
     */
    showErrorOverlay(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'map-error-overlay';
        errorDiv.innerHTML = `
            <div class="alert alert-danger m-3">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
        
        document.getElementById(this.mapId).appendChild(errorDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
    
    /**
     * Toggle the visibility of ward boundaries
     */
    toggleBoundaries() {
        // Check if boundaries exist and are visible
        if (this.wardBoundaries) {
            if (this.map.hasLayer(this.wardBoundaries)) {
                this.map.removeLayer(this.wardBoundaries);
            } else {
                this.wardBoundaries.addTo(this.map);
            }
            return true;
        }
        return false;
    }
    
    /**
     * Toggle the heatmap view of crime data
     */
    toggleHeatmap() {
        // Check if heatmap exists
        if (this.heatmapLayer) {
            // Remove existing heatmap
            this.map.removeLayer(this.heatmapLayer);
            this.heatmapLayer = null;
            return true;
        }
        
        // Get current hour from slider
        const hour = parseInt(document.getElementById('time-slider').value);
        
        // Fetch crime data for this hour to generate heatmap
        fetch(`/api/predict?hour=${hour}`)
            .then(response => response.json())
            .then(data => {
                // Create heatmap data points
                const heatPoints = [];
                
                // For each ward, create a heat point based on crime probability
                data.wards.forEach(ward => {
                    // Get ward marker for coordinates
                    const marker = this.wardMarkers[ward.ward_id];
                    if (marker) {
                        const latlng = marker.getLatLng();
                        const intensity = ward.crime_probability * 50; // Scale for visibility
                        heatPoints.push([latlng.lat, latlng.lng, intensity]);
                    }
                });
                
                // If no heatmap plugin is available, show error
                if (!L.heatLayer) {
                    this.showErrorOverlay('Heatmap plugin not available. Please refresh the page.');
                    return;
                }
                
                // Create and add heatmap layer
                this.heatmapLayer = L.heatLayer(heatPoints, {
                    radius: 35,
                    blur: 35,
                    maxZoom: 15,
                    max: 1.0,
                    gradient: {
                        0.3: 'green',
                        0.5: 'yellow',
                        0.7: 'orange',
                        1.0: 'red'
                    }
                }).addTo(this.map);
                
                return true;
            })
            .catch(error => {
                console.error('Error creating heatmap:', error);
                this.showErrorOverlay('Failed to create heatmap. Please try again.');
                return false;
            });
    }
    
    // Expose the safetyMap instance to the global scope for access in index.html
    static initialize(mapId) {
        window.safetyMap = new SafetyMap(mapId);
        window.safetyMap.init();
        return window.safetyMap;
    }
}
