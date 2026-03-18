document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("aqiForm");
    const predictBtn = document.getElementById("predictBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    
    const resultsSection = document.getElementById("resultsSection");
    const placeholder = document.getElementById("placeholder");
    const resultsContent = document.getElementById("resultsContent");
    
    const aqiCircle = document.getElementById("aqiCircle");
    const aqiValue = document.getElementById("aqiValue");
    const aqiCategory = document.getElementById("aqiCategory");
    const aqiSuggestion = document.getElementById("aqiSuggestion");

    let chartInstance = null;

    form.addEventListener("submit", async async (e) => {
        e.preventDefault();
        
        // Setup UI for Loading
        btnText.textContent = "Predicting...";
        btnSpinner.classList.remove("d-none");
        predictBtn.disabled = true;

        // Gather Data
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value);
        }

        try {
            // API Call
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                displayResults(result);
                renderChart(result.input);
            } else {
                alert("Error: " + (result.error || "Failed to predict AQI"));
            }
        } catch (error) {
            console.error("Error submitting form:", error);
            alert("Network error or server unavailable.");
        } finally {
            // Restore UI
            btnText.textContent = "Predict AQI";
            btnSpinner.classList.add("d-none");
            predictBtn.disabled = false;
        }
    });

    function displayResults(result) {
        // Hide placeholder, show content
        placeholder.classList.add("d-none");
        resultsContent.classList.remove("d-none");
        
        // Remove old classes from circle and category text
        aqiCircle.className = "aqi-circle fade-in my-4 mx-auto";
        aqiCategory.className = "fw-bold mb-3";

        // Map Category to CSS classes
        let catClassSuffix = "";
        const catMap = {
            "Good": "good",
            "Satisfactory": "satisfactory",
            "Moderate": "moderate",
            "Poor": "poor",
            "Very Poor": "very-poor",
            "Severe": "severe"
        };
        
        catClassSuffix = catMap[result.category] || "moderate";

        aqiCircle.classList.add(`aqi-${catClassSuffix}`);
        aqiCategory.classList.add(`text-${catClassSuffix}`);

        // Animate AQI Value counting up
        animateValue(aqiValue, 0, result.aqi, 1000);
        
        aqiCategory.textContent = result.category;
        aqiSuggestion.textContent = result.suggestion;
    }

    function renderChart(inputData) {
        const ctx = document.getElementById('pollutantChart').getContext('2d');
        
        // Destroy old chart if exists
        if (chartInstance) {
            chartInstance.destroy();
        }

        const labels = Object.keys(inputData);
        const data = Object.values(inputData);

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Pollutant Breakdown',
                    data: data,
                    backgroundColor: 'rgba(79, 172, 254, 0.6)',
                    borderColor: 'rgba(79, 172, 254, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#c9d1d9'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#8b949e'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#8b949e'
                        }
                    }
                }
            }
        });
    }

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
