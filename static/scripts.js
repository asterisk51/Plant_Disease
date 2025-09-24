function trackDetection() {
    // Get current count from localStorage
    let count = parseInt(localStorage.getItem('imageDetectionCount') || '0');
    count++; // increment by 1
    localStorage.setItem('imageDetectionCount', count);

    // Check if it reached 3
    if (count >= 3) {
        showSignupWarning();
    }
}

function showSignupWarning() {
    const proceed = confirm(
        "You've reached your 3 free image detections. Click OK to sign up and continue using the service."
    );
    if (proceed) {
        redirectToSignup();
    }
}

function redirectToSignup() {
    // Replace with your actual signup page URL
    window.location.href = '/signup';
}

localStorage.setItem('imageDetectionCount', '0');
