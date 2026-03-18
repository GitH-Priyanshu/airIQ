document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Date/Time in Navbar
    const updateTime = () => {
        const now = new Date();
        document.getElementById('currentDate').textContent = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
        document.getElementById('currentTime').textContent = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    };
    setInterval(updateTime, 1000);
    updateTime();

    // Chart.js Setup
    Chart.defaults.color = '#8b949e';
    Chart.defaults.font.family = "'Poppins', sans-serif";
    
    let scatterChartInstance = null;
    
    // Fetch Real ML Metrics on Load
    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            if(data.error) {
                console.error(data.error);
                return;
            }
            populateMetricsTable(data.metrics);
            drawScatterChart(data.scatter);
        })
        .catch(err => console.error("Error loading metrics:", err));

    function populateMetricsTable(metrics) {
        const tbody = document.querySelector('.performance-table tbody');
        tbody.innerHTML = '';
        
        metrics.forEach(m => {
            const tr = document.createElement('tr');
            if(m.is_best) {
                tr.className = 'best-model-row';
                tr.setAttribute('data-bs-toggle', 'tooltip');
                tr.setAttribute('title', 'Best performing model');
                tr.innerHTML = `
                    <td class="fw-bold text-warning"><i class="fa-solid fa-crown me-1"></i> ${m.name}</td>
                    <td class="text-center font-monospace">${m.mae}</td>
                    <td class="text-center font-monospace">${m.rmse}</td>
                    <td class="text-end pe-3 fw-bold text-warning font-monospace">${m.r2} (Best)</td>
                `;
            } else {
                tr.innerHTML = `
                    <td>${m.name}</td>
                    <td class="text-center font-monospace">${m.mae}</td>
                    <td class="text-center font-monospace">${m.rmse}</td>
                    <td class="text-end pe-3 font-monospace">${m.r2}</td>
                `;
            }
            tbody.appendChild(tr);
        });
    }

    function drawScatterChart(scatterData) {
        const ctxScatter = document.getElementById('scatterChart').getContext('2d');
        if(scatterChartInstance) scatterChartInstance.destroy();

        scatterChartInstance = new Chart(ctxScatter, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Real Predictions (Hybrid)',
                        data: scatterData,
                        backgroundColor: '#BA68C8',
                        borderColor: 'rgba(186, 104, 200, 0.8)',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: 'Perfect Prediction',
                        data: [{x: 0, y: 0}, {x: 450, y: 450}],
                        type: 'line',
                        borderColor: '#4CAF50',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#c9d1d9' } },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                if(ctx.datasetIndex === 1) return 'Perfect Prediction';
                                return `Actual: ${ctx.raw.x}, Predicted: ${ctx.raw.y.toFixed(0)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear', position: 'bottom',
                        title: { display: true, text: 'Actual AQI (Ground Truth)', color: '#8b949e' },
                        grid: { color: 'rgba(255,255,255,0.05)' }, max: 450
                    },
                    y: {
                        title: { display: true, text: 'Predicted AQI', color: '#8b949e' },
                        grid: { color: 'rgba(255,255,255,0.05)' }, max: 450
                    }
                }
            }
        });
    }

    // Static Bar Chart
    const ctxBar = document.getElementById('barChart').getContext('2d');
    new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: ['CO', 'NO2', 'OZONE', 'PM10'],
            datasets: [{
                label: 'Relative Contribution',
                data: [1.5, 45, 35, 180],
                backgroundColor: ['rgba(255, 193, 7, 0.8)', 'rgba(76, 175, 80, 0.8)', 'rgba(33, 150, 243, 0.8)', 'rgba(244, 67, 54, 0.8)'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    // 6. What-If Simulation Panel Logic (REAL API CALL)
    const sliders = ['simCO', 'simNO2', 'simPM10', 'simOzone'];
    sliders.forEach(id => {
        document.getElementById(id).addEventListener('input', (e) => {
            const valSpan = id.replace('sim', '').toLowerCase() + 'Val';
            document.getElementById(valSpan).textContent = e.target.value;
        });
    });

    document.getElementById('simulateBtn').addEventListener('click', async () => {
        const payload = {
            PM10: parseInt(document.getElementById('simPM10').value),
            NO2: parseInt(document.getElementById('simNO2').value),
            OZONE: parseInt(document.getElementById('simOzone').value),
            CO: parseFloat(document.getElementById('simCO').value)
        };
        
        const btn = document.getElementById('simulateBtn');
        btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
        btn.disabled = true;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if(data.success) {
                const simAqi = data.aqi;
                const catText = data.category;
                
                let colorClass = "";
                if(simAqi <= 50) colorClass = "text-success";
                else if(simAqi <= 100) colorClass = "text-warning";
                else if(simAqi <= 200) { colorClass = "text-orange"; document.getElementById('simResultDisplay').style.color = '#fd7e14'; }
                else if(simAqi <= 300) colorClass = "text-danger";
                else colorClass = "text-danger text-decoration-underline";

                const displayElem = document.getElementById('simResultDisplay');
                displayElem.innerHTML = `Predicted AQI: ${simAqi} <span class="fs-6 fw-normal ${colorClass}">(${catText})</span>`;
                
                displayElem.classList.remove('fade-in');
                void displayElem.offsetWidth;
                displayElem.classList.add('fade-in');
            } else {
                alert("Prediction failed: " + data.error);
            }
        } catch (err) {
            console.error(err);
            alert("Failed to connect to ML Backend.");
        } finally {
            btn.innerHTML = 'Simulate AQI';
            btn.disabled = false;
        }
    });

    // 7. City Table Populator
    const tableData = [
        { city: 'Delhi', aqi: 310, cat: 'Very Poor', catClass: 'bg-city-very-poor' },
        { city: 'Dehradun', aqi: 120, cat: 'Moderate', catClass: 'bg-city-moderate' },
        { city: 'Bangalore', aqi: 55, cat: 'Satisfactory', catClass: 'bg-city-good' },
        { city: 'Mumbai', aqi: 145, cat: 'Moderate', catClass: 'bg-city-moderate' },
        { city: 'Chennai', aqi: 48, cat: 'Good', catClass: 'bg-city-good' }
    ];

    const tbody = document.getElementById('cityTableBody');
    tableData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="ps-3 fw-semibold">${row.city}</td>
            <td class="text-center font-monospace">${row.aqi}</td>
            <td class="text-end pe-3"><span class="badge ${row.catClass} px-2 py-1">${row.cat}</span></td>
        `;
        tbody.appendChild(tr);
    });

    // 9. Future Features Notification Mapping
    const futureBtns = document.querySelectorAll('.future-feature-btn');
    futureBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            alert("This feature will be implemented in future versions.");
            // alternatively you can use a Bootstrap modal, but prompt asked for a popup. 
        });
    });

    // Helper: Count up animation
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
