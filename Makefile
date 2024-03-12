
streamlit:
	@streamlit run app.py

reinstall_package:
	@pip uninstall -y power || :
	@pip install -e .
