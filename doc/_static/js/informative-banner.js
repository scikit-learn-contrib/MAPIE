document.addEventListener('DOMContentLoaded', function() {
    var path = window.location.pathname;

    if (path.includes('/latest/') || path.includes('/stable/')) {
        document.getElementById('version-banner').style.display = 'block';
    }
});