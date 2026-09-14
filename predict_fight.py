from src.predictor import predict_fight_outcome, generate_report

fighter_a = input("Enter the name of Fighter A: ")
fighter_b = input("Enter the name of Fighter B: ")

result = predict_fight_outcome(fighter_a, fighter_b)
report = generate_report(result)
with open("predictions/latest_predictions.txt", "w") as f:
    f.write(report)

print(report)