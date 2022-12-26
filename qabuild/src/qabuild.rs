// Build training and evaluation data sets for QA research.

use ::function_name::named;
use rand::Rng;
use serde::Serialize;
use serde_json::{Deserializer, Value};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::rc::Rc;

// These are possible answers to a question.  Each answer includes a substring
// from the reading passage (context) and its starting offset in the passage.
#[derive(Clone, Serialize)]
struct Answers {
    text: Vec<String>,
    answer_start: Vec<i64>,
}

// This is a flattened example that we use to create a dataset of training
// examples readable by the ELECTRA-small model used in the NLP course project.
// We will hash to one of these with an id.
#[derive(Clone)]
struct Example {
    title: Rc<String>,
    context: Rc<String>,
    question: String,
    answers: Answers,
}

#[derive(Serialize)]
struct Output {
    title: Vec<String>,
    context: Vec<String>,
    question: Vec<String>,
    id: Vec<String>,
    answers: Vec<Answers>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 5 {
        eprintln!(
            "Usage: {} FILE [APPEND-FILE PREPEND-FILE TWOWAY-FILE]",
            args[0]
        );
        eprintln!(
            "Example: {} train-convHighConf-pretty.json train-appended.json",
            args[0]
        );
        std::process::exit(1);
    }

    let raw_examples = read_raw_examples(&args[1]).unwrap();

    if args.len() > 2 {
        let append_examples = get_append_examples(&raw_examples);
        write_training_examples(append_examples, &args[2]).unwrap();
        if args.len() > 3 {
            let twoway_examples = get_twoway_examples(&raw_examples);
            write_training_examples(twoway_examples, &args[3]).unwrap();
        }
    }

    // println!("{:?} {:?}", raw_examples.len(), append_examples.len());
}

#[named]
fn get_append_examples(raw_examples: &HashMap<String, Example>) -> HashMap<String, Example> {
    let mut append_examples: HashMap<String, Example> = HashMap::<String, Example>::new();

    let mut clean_examples = 0;
    let mut appended_examples = 0;
    let mut rng = rand::thread_rng();

    for (k, v) in raw_examples {
        if !k.contains('-') {
            let altid = k.to_string() + "-high-conf";
            if !raw_examples.contains_key(&altid) {
                append_examples.insert(k.to_string(), v.clone());
                clean_examples += 1;
            }
        } else {
            let mut tokens = k.split('-');
            let baseid = tokens.next().unwrap();
            if raw_examples.contains_key(baseid) {
                let sample = rng.gen_range(0..69808);
                if sample < 26009 {
                    let v2 = &raw_examples[baseid];
                    append_examples.insert(baseid.to_string(), v2.clone());
                    clean_examples += 1;
                } else {
                    append_examples.insert(baseid.to_string(), v.clone());
                    appended_examples += 1;
                }
            }
        }
    }

    println!(
        "{}: clean_examples: {}, appended_examples: {}",
        function_name!(),
        clean_examples,
        appended_examples
    );

    append_examples
}

#[named]
fn get_twoway_examples(raw_examples: &HashMap<String, Example>) -> HashMap<String, Example> {
    let mut twoway_examples: HashMap<String, Example> = HashMap::<String, Example>::new();

    let mut clean_examples = 0;
    let mut appended_examples = 0;
    let mut prepended_examples = 0;

    let mut rng = rand::thread_rng();

    for (k, v) in raw_examples {
        if !k.contains('-') {
            let altid = k.to_string() + "-high-conf";
            if !raw_examples.contains_key(&altid) {
                twoway_examples.insert(k.to_string(), v.clone());
                clean_examples += 1;
            }
        } else {
            let mut tokens = k.split('-');
            let baseid = tokens.next().unwrap();
            if raw_examples.contains_key(baseid) {
                let sample = rng.gen_range(0..69808);
                if sample < 11339 {
                    // Add a clean example.
                    let v2 = &raw_examples[baseid];
                    twoway_examples.insert(baseid.to_string(), v2.clone());
                    clean_examples += 1;
                } else if sample < 40469 {
                    // Add an appended example.
                    twoway_examples.insert(baseid.to_string(), v.clone());
                    appended_examples += 1;
                } else {
                    // Add a prepended example.
                    let v2 = &raw_examples[baseid];
                    let mut last_sentence = v.context[v2.context.len()..].trim().to_string();
                    last_sentence += " ";
                    let start_offset = last_sentence.len();
                    let mut answers = v2.answers.clone();
                    for start_pos in &mut answers.answer_start {
                        *start_pos += start_offset as i64;
                    }
                    twoway_examples.insert(
                        baseid.to_string(),
                        Example {
                            title: v2.title.clone(),
                            context: Rc::<String>::new(last_sentence + &v2.context),
                            question: v2.question.clone(),
                            answers: answers,
                        },
                    );
                    prepended_examples += 1;
                }
            }
        }
    }

    println!(
        "{}: clean_examples: {}, appended_examples: {}, prepended_examples: {}",
        function_name!(),
        clean_examples,
        appended_examples,
        prepended_examples
    );

    twoway_examples
}

fn read_raw_examples<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Example>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut stream = Deserializer::from_reader(reader).into_iter::<Value>();

    let value = stream.next().unwrap();
    let binding = value?;
    let data = binding["data"].as_array().unwrap();
    // println!("{:?}", data.len());

    let mut raw_examples = HashMap::<String, Example>::new();

    for pargroup in data {
        let title = Rc::new(pargroup["title"].as_str().unwrap().to_string());

        let paragraphs = pargroup["paragraphs"].as_array().unwrap();
        for paragraph in paragraphs {
            let context = Rc::new(paragraph["context"].as_str().unwrap().to_string());

            let qas = paragraph["qas"].as_array().unwrap();
            for qa in qas {
                let question = qa["question"].as_str().unwrap().to_string();
                let id = qa["id"].as_str().unwrap().to_string();
                let mut answer_start: Vec<i64> = vec![];
                let mut answer_text: Vec<String> = vec![];
                let answers = qa["answers"].as_array().unwrap();
                for answer in answers {
                    answer_start.push(answer["answer_start"].as_i64().unwrap());
                    let atext = answer["text"].as_str().unwrap();
                    answer_text.push(atext.to_string());
                }
                raw_examples.insert(
                    id,
                    Example {
                        title: title.clone(),
                        context: context.clone(),
                        question: question,
                        answers: Answers {
                            answer_start: answer_start,
                            text: answer_text,
                        },
                    },
                );
            }
        }
    }

    Ok(raw_examples)
}

fn write_training_examples<P: AsRef<Path>>(
    examples: HashMap<String, Example>,
    path: P,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let mut output = Output {
        title: vec![],
        context: vec![],
        question: vec![],
        id: vec![],
        answers: vec![],
    };

    // Generate data structure corresponding to flattened output.
    for (k, v) in examples {
        output.title.push(v.title.to_string());
        output.context.push(v.context.to_string());
        output.question.push(v.question);
        output.id.push(k);
        output.answers.push(v.answers);
    }

    let mut data = HashMap::<String, Output>::new();
    data.insert("data".to_string(), output);

    serde_json::to_writer_pretty(writer, &data)?;

    Ok(())
}
