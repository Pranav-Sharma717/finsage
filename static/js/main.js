// Form handling for personal guidance
document.addEventListener('DOMContentLoaded', () => {
    const guidanceForm = document.getElementById('guidance-form');
    if (guidanceForm) {
        guidanceForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(guidanceForm);
            const submitButton = guidanceForm.querySelector('button[type="submit"]');
            const loadingSpinner = document.createElement('div');
            loadingSpinner.className = 'loading';
            
            try {
                // Show loading state
                submitButton.disabled = true;
                submitButton.appendChild(loadingSpinner);
                
                const response = await fetch('/personal-guidance', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Display recommendations
                displayRecommendations(data);
            } catch (error) {
                console.error('Error:', error);
                showError('An error occurred while processing your request.');
            } finally {
                // Reset button state
                submitButton.disabled = false;
                loadingSpinner.remove();
            }
        });
    }
});

// Display recommendations in a card layout
function displayRecommendations(data) {
    const recommendationsContainer = document.getElementById('recommendations');
    if (!recommendationsContainer) return;
    
    recommendationsContainer.innerHTML = '';
    
    Object.entries(data).forEach(([category, suggestion]) => {
        const card = document.createElement('div');
        card.className = 'card mb-4';
        
        card.innerHTML = `
            <h3 class="text-xl font-bold text-blue-400 mb-2">${formatCategory(category)}</h3>
            <p class="text-gray-400">${suggestion}</p>
        `;
        
        recommendationsContainer.appendChild(card);
    });
    
    // Animate cards
    const cards = recommendationsContainer.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

// Format category names for display
function formatCategory(category) {
    return category
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'bg-red-500 text-white p-4 rounded-lg mb-4';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);
    
    // Remove error message after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Smooth scroll to sections
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add hover effects to cards
document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.classList.add('hover-glow');
    });
    
    card.addEventListener('mouseleave', () => {
        card.classList.remove('hover-glow');
    });
}); 