import openai

api_key = input("Enter OpenAI API key:")
openai.api_key = api_key

# Upload datasets
with open("train.jsonl", "rb") as train_file:
    train_response = openai.Dataset.create(
        file=train_file,
        purpose="fine-tuning",
    )

with open("valid.jsonl", "rb") as valid_file:
    valid_response = openai.Dataset.create(
        file=valid_file,
        purpose="fine-tuning",
    )

# Fine-tune the model
model = "text-davinci-002" 
train_dataset_id = train_response["id"]
valid_dataset_id = valid_response["id"]

fine_tuning_config = {
  "model": model,
  "dataset": {
    "train": train_dataset_id,
    "valid": valid_dataset_id,
  },
  "n_epochs": 3,
  "batch_size": 4,
  "max_grad_norm": 1.0,
  "learning_rate": 3e-5,
  "weight_decay": 0.01,
}

fine_tuning_response = openai.FineTuning.create(**fine_tuning_config)

# Use the fine-tuned model for inference
fine_tuned_model_id = fine_tuning_response["id"]

def generate_unit_test(code):
    prompt = f"Generate C# unit tests for the following C# code:\n{csharp_code}\n\nUnit tests:\n"
    response = openai.Completion.create(
        engine=fine_tuned_model_id,
        prompt=prompt,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"].strip()

# Example usage
csharp_code = '''
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
namespace MVC_BasicTutorials.Controllers
{
    public class StudentController : Controller
    {
        // GET: Student
        public string Index()
        {
                return "This is Index action method of StudentController";
        }
    }
}
'''
UnitTest = generate_unit_test(csharp_code)
print(f"UnitTests: {}")
