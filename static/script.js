document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileRes = document.getElementById('file-res');
    const resetBtn = document.getElementById('reset-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const resultsSection = document.getElementById('results-section');
    const errorBanner = document.getElementById('error-banner');

    // Elemen UI hasil analisis
    const confidenceCircle = document.getElementById('confidence-circle');
    const confidenceText = document.getElementById('confidence-text');
    const aiProbBar = document.getElementById('ai-prob-bar');
    const authProbBar = document.getElementById('auth-prob-bar');
    const aiProbText = document.getElementById('ai-prob-text');
    const authProbText = document.getElementById('auth-prob-text');
    const statusBadge = document.getElementById('status-badge');
    const summaryText = document.getElementById('summary-text');

    // File yang sedang dipilih (dikirim ke backend saat "Analisa Gambar" ditekan)
    let selectedFile = null;

    // Drag & drop
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Fungsi untuk menangani file yang diunggah
    function handleFile(file) {
      if (!file.type.startsWith('image/')) return;

      selectedFile = file;

      const reader = new FileReader();

      reader.onload = function (e) {

        imagePreview.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

        const img = new Image();

        img.onload = function () {
            fileRes.textContent =
                `${this.naturalWidth} × ${this.naturalHeight}px`;
        };

        img.src = e.target.result;

        // <<< BAGIAN INI JANGAN DIHAPUS
        uploadArea.style.display = 'none';
        previewArea.classList.add('is-visible');
        resultsSection.style.display = 'none';
        resultsSection.classList.remove('visible');
        hideError();
      };

      reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
      uploadArea.style.display = 'flex';
      previewArea.classList.remove('is-visible');
      fileInput.value = '';
      selectedFile = null;
      resultsSection.style.display = 'none';
      resultsSection.classList.remove('visible');
      hideError();
    });

    analyzeBtn.addEventListener('click', () => {
      if (!selectedFile) return;

      // Status loading
      btnText.style.display = 'none';
      btnSpinner.style.display = 'block';
      analyzeBtn.disabled = true;
      resultsSection.style.display = 'none';
      resultsSection.classList.remove('visible');
      hideError();

      const formData = new FormData();
      formData.append('image', selectedFile);

      fetch('/predict', {
          method: 'POST',
          body: formData,
      })
          .then(async (response) => {
              let data;
              try {
                  data = await response.json();
              } catch (parseErr) {
                  throw new Error('Respons server tidak valid.');
              }
              if (!response.ok || !data.success) {
                  throw new Error(data.error || 'Terjadi kesalahan saat menganalisa gambar.');
              }
              return data;
          })
          .then((data) => {
              showResults(data);
          })
          .catch((err) => {
              showError(err.message);
          })
          .finally(() => {
              btnText.style.display = 'block';
              btnSpinner.style.display = 'none';
              analyzeBtn.disabled = false;
          });
    });

    function showError(message) {
      errorBanner.textContent = message;
      errorBanner.classList.add('is-visible');
    }

    function hideError() {
      errorBanner.textContent = '';
      errorBanner.classList.remove('is-visible');
    }

    // Menampilkan hasil analisis dari backend.
    // `data` adalah hasil JSON dari endpoint /predict:
    //   { is_ai_generated, confidence, probability_ai, probability_real, ... }
    function showResults(data) {
      const isAI = data.is_ai_generated;
      const score = data.confidence;              // confidence kelas yang diprediksi
      const aiProb = data.probability_ai;
      const realProb = data.probability_real;

      resultsSection.style.display = 'grid';

      // Memicu reflow agar animasi transisi berjalan
      void resultsSection.offsetWidth;

      resultsSection.classList.add('visible');

      if (isAI) {
          statusBadge.classList.remove('is-authentic');
          statusBadge.innerHTML = `<span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 1;">warning</span> <h2>Gambar Hasil AI</h2>`;
          summaryText.textContent = 'Gambar tersebut kemungkinan besar dihasilkan oleh AI berdasarkan pola artefak yang terdeteksi oleh model Vision Transformer (ViT-Tiny).';

          confidenceCircle.classList.remove('stroke-primary');
          confidenceCircle.classList.add('stroke-error');
      } else {
          statusBadge.classList.add('is-authentic');
          statusBadge.innerHTML = `<span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 1;">check_circle</span> <h2>Gambar Asli</h2>`;
          summaryText.textContent = 'Gambar tersebut kemungkinan besar adalah gambar Asli. Model Vision Transformer (ViT-Tiny) mendeteksi pola noise alami yang konsisten dengan sensor kamera.';

          confidenceCircle.classList.remove('stroke-error');
          confidenceCircle.classList.add('stroke-primary');
      }

      aiProbBar.style.width = `${aiProb}%`;
      authProbBar.style.width = `${realProb}%`;
      aiProbText.textContent = `${aiProb}%`;
      authProbText.textContent = `${realProb}%`;

      confidenceCircle.style.strokeDasharray = `${score}, 100`;
      animateValue(confidenceText, 0, score, 1000);
    }

    function animateValue(obj, start, end, duration) {
      let startTimestamp = null;
      const step = (timestamp) => {
          if (!startTimestamp) startTimestamp = timestamp;
          const progress = Math.min((timestamp - startTimestamp) / duration, 1);
          const current = progress * (end - start) + start;
          obj.innerHTML = current.toFixed(1) + '%';
          if (progress < 1) {
              window.requestAnimationFrame(step);
          }
      };
      window.requestAnimationFrame(step);
    }
});
