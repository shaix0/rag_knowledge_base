// quiz.js
document.addEventListener('DOMContentLoaded', () => {
    if (!window.quizData || !quizData.length) return;

    const form = document.getElementById('quiz-form');
    const submitArea = document.getElementById('submit-area');
    const resultSummary = document.getElementById('result-summary');
    const navigatorDiv = document.getElementById('question-navigator');
    const customAlert = document.getElementById('custom-alert');
    const totalQuestions = quizData.length;
    TEMP_QUIZ_RESULTS = {}

    // 計時器
    let timeRemaining = window.timeLimit * 60;
    let timerInterval = null;

    function formatTime(sec) {
        const minutes = Math.floor(sec / 60);
        const seconds = sec % 60;
        return `${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}`;
    }

    function showAlert(message) {
        if (!customAlert) return;
        customAlert.innerHTML = `<strong>注意！</strong> ${message}`;
        customAlert.style.display = 'block';
        setTimeout(() => { customAlert.style.display = 'none'; }, 5000);
    }

    function forceSubmit() {
        clearInterval(timerInterval);
        showAlert('時間到！測驗已強制交卷。');
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn && !submitBtn.disabled) submitBtn.click();
    }

    function startTimer() {
        const timerDisplay = document.getElementById('timer-display');
        if (!timerDisplay) return;

        timerDisplay.textContent = `⏱ ${formatTime(timeRemaining)}`;
        timerDisplay.classList.remove('alert-time');

        timerInterval = setInterval(() => {
            if (timeRemaining <= 0) {
                forceSubmit();
            } else {
                timeRemaining--;
                timerDisplay.textContent = `⏱ ${formatTime(timeRemaining)}`;
                if (timeRemaining <= timeRemaining * 0.1) {
                    timerDisplay.classList.add('alert-time');
                }
            }
        }, 1000);
    }

    // 導覽列
    function generateNavigator() {
        if (!navigatorDiv) return;
        navigatorDiv.innerHTML = '';
        for (let i = 1; i <= totalQuestions; i++) {
            const navItem = document.createElement('div');
            navItem.classList.add('nav-item');
            const navLink = document.createElement('a');
            navLink.classList.add('nav-link-q');
            navLink.textContent = i;
            navLink.id = `nav-q-${i}`;
            navLink.dataset.qIndex = i;
            navLink.addEventListener('click', () => {
                const targetEl = document.getElementById(`q-${i}`);
                if (targetEl) targetEl.scrollIntoView({behavior:'smooth', block:'start'});
            });
            navItem.appendChild(navLink);
            navigatorDiv.appendChild(navItem);
        }
    }

    function updateNavigator(qIndex, answered=false, isResultMode=false, isCorrect=false) {
        const navLink = document.getElementById(`nav-q-${qIndex}`);
        if (!navLink) return;
        navLink.classList.remove('answered','correct-result','incorrect-result');
        if (isResultMode) {
            navLink.classList.add(isCorrect ? 'correct-result' : 'incorrect-result');
        } else if (answered) {
            navLink.classList.add('answered');
        }
    }

    function setupAnswerListener() {
        const radios = form.querySelectorAll('input[type="radio"]');
        radios.forEach(radio => {
            radio.addEventListener('change', function() {
                updateNavigator(this.dataset.qIndex, true);
            });
            if (radio.checked) updateNavigator(radio.dataset.qIndex, true);
        });
    }

    function displayResults(result) {
        clearInterval(timerInterval);

        document.getElementById('final-score').textContent = result.total_score;
        document.getElementById('max-score').textContent = result.total_questions * result.score_per_question;
        document.getElementById('correct-count-summary').textContent = result.correct_count;
        document.getElementById('incorrect-count-summary').textContent = result.total_questions - result.correct_count;

        if (submitArea) submitArea.classList.add('d-none');
        if (resultSummary) resultSummary.classList.remove('d-none');

        if (result.detailed_results) {
            result.detailed_results.forEach(item => {
                const qIndex = item.question_index;
                const isCorrect = item.is_correct;
                const correctOption = item.correct_answer;

                const questionCard = document.getElementById(`q-${qIndex}`);
                if (!questionCard) return;

                const footer = questionCard.querySelector('.result-footer');
                const radios = questionCard.querySelectorAll('input[type="radio"]');

                radios.forEach(radio => {
                    radio.disabled = true;
                    const label = radio.nextElementSibling;
                    label.classList.remove('correct-option','user-incorrect-choice');

                    if (radio.value === correctOption) label.classList.add('correct-option');
                    if (radio.checked && !isCorrect) label.classList.add('user-incorrect-choice');
                });

                if (footer) {
                    footer.innerHTML = `<div><strong>標準答案：</strong> <span class="text-success">${correctOption}</span></div>`;
                    footer.classList.remove('d-none');
                }

                updateNavigator(qIndex,true,true,isCorrect);
            });
        }
    }

    // 表單提交
    form.addEventListener('submit', async (e)=>{
        e.preventDefault();
        clearInterval(timerInterval);

        const userAnswers = {};
        quizData.forEach((_, index)=>{
            const qIndex = index+1;
            const radios = document.getElementsByName(`question-${qIndex}`);
            for(const r of radios){
                if(r.checked){ userAnswers[`question-${qIndex}`]=r.value; break; }
            }
        });

        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled=true; submitBtn.textContent='評分中...';

        try {
            const resp = await fetch(window.submitQuizUrl, {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({
                    answers: userAnswers,
                    quiz_data: quizData.map(q=>q.id),
                    time_spent: window.timeLimit*60 - timeRemaining
                })
            });

            const data = await resp.json();
            if(!resp.ok || !data.success) throw new Error(data.message || '伺服器錯誤');

            displayResults(data);  // 保留原本 displayResults 功能
            document.getElementById('view-report-btn').href = '/report_page';

        } catch(err){
            console.error(err);
            showAlert('評分時發生錯誤: '+err.message);
            if(timeRemaining>0){ submitBtn.disabled=false; submitBtn.textContent='送出測驗'; startTimer(); }
        }
    });

    // 初始化
    generateNavigator();
    setupAnswerListener();
    startTimer();
});