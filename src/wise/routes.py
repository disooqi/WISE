from . import app
from .forms import SearchForm
from flask import render_template, flash, redirect, url_for, request, session
from .wise import Wise


@app.route('/', methods=['GET', 'POST'])
def home():
    form = SearchForm()

    if form.validate_on_submit():
        return redirect(url_for('result', question=form.question.data, num=form.n_answers.data))
    return render_template('index.html', form=form)


@app.route('/result', methods=['GET', 'POST'])
def result():
    form = SearchForm()
    if request.method == 'GET':
        question = request.args.get('question')
        n_answers = int(request.args.get('num'))

        form.question.data = question
        form.n_answers.data = n_answers
        session['n_answers'] = n_answers
        WISE = Wise()
        answers = WISE.ask(question_text=question, n_max_answers=n_answers, merge_answers=False)
        return render_template('result.html', question=question, answers=answers, form=form)
    elif form.validate_on_submit():
        return redirect(url_for('result', question=form.question.data, num=session['n_answers']))
    else:
        return render_template('result.html', question=form.question.data, form=form)
