
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");

if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
        navLinks.classList.toggle("open");
    });

    navLinks.querySelectorAll("a").forEach(link => {
        link.addEventListener("click", () => {
            navLinks.classList.remove("open");
        });
    });
}


// Password visibility
document.querySelectorAll(".toggle-password").forEach(button => {
    button.addEventListener("click", () => {
        const field = document.getElementById(button.dataset.target);

        if (!field) return;

        if (field.type === "password") {
            field.type = "text";
            button.textContent = "Hide";
        } else {
            field.type = "password";
            button.textContent = "Show";
        }
    });
});

// Registration form validation with real Flask submission
const registerForm = document.getElementById("registerForm");

if (registerForm) {
    registerForm.addEventListener("submit", event => {
        const password = document.getElementById("registerPassword").value;
        const confirm = document.getElementById("confirmPassword").value;
        const message = document.getElementById("registerMessage");

        if (password.length < 8) {
            event.preventDefault();
            message.textContent = "Password must contain at least 8 characters.";
            return;
        }

        if (password !== confirm) {
            event.preventDefault();
            message.textContent = "Passwords do not match.";
            return;
        }

        // Allow normal POST submission to Flask /register
        message.textContent = "Creating your account...";
    });
}

// Login form - allow real Flask backend submission
const loginForm = document.getElementById("loginForm");

if (loginForm) {
    loginForm.addEventListener("submit", event => {
        const message = document.getElementById("loginMessage");

        if (message) {
            message.textContent = "Signing in...";
        }

        // Allow normal POST submission to Flask /login
    });
}


// NeuroScan AI Dashboard
console.log("NeuroScan AI Dashboard loaded successfully.");


// MRI Upload Page - Image Preview
document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("mriFile");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const fileName = document.getElementById("fileName");
    const uploadForm = document.getElementById("mriUploadForm");
    const uploadMessage = document.getElementById("uploadMessage");

    if (!fileInput) return;

    fileInput.addEventListener("change", function () {
        const file = this.files[0];

        uploadMessage.textContent = "";

        if (!file) return;

        const allowedTypes = ["image/jpeg", "image/png"];

        if (!allowedTypes.includes(file.type)) {
            alert("Please select a JPG, JPEG, or PNG image.");
            fileInput.value = "";
            previewContainer.style.display = "none";
            fileName.textContent = "";
            return;
        }

        imagePreview.src = URL.createObjectURL(file);
        previewContainer.style.display = "block";
        fileName.textContent = "Selected file: " + file.name;
    });


    uploadForm.addEventListener("submit", function (event) {
        if (!fileInput.files.length) {
            event.preventDefault();
            uploadMessage.textContent = "Please select an MRI image first.";
            return;
        }

        uploadMessage.textContent = "Uploading image...";
    });
});
