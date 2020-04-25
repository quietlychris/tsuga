
use std::fs;

fn main() {
    let MNIST_DIR = "../../data/mnist";

    let TRAIN_SET_SIZE = 400;
    let TRAIN_DIR = MNIST_DIR.to_owned() + "/" + "train";
    let CONSOLIDATED_TRAIN_DIR = MNIST_DIR.to_owned() + "/" + "train_subset";
    fs::remove_dir_all(&CONSOLIDATED_TRAIN_DIR).expect("Couldn't remove previous consolidated training directory");
    fs::create_dir_all(&CONSOLIDATED_TRAIN_DIR).expect("Couldn't build the consolidated training directory");

    let TEST_SET_SIZE=100;
    let TEST_DIR = MNIST_DIR.to_owned() + "/" + "test";
    let CONSOLIDATED_TEST_DIR = MNIST_DIR.to_owned() + "/" + "test_subset";
    fs::remove_dir_all(&CONSOLIDATED_TEST_DIR).expect("Couldn't remove previous consolidated training directory");
    fs::create_dir_all(&CONSOLIDATED_TEST_DIR).expect("Couldn't build the consolidated testing directory");


    // Let's accumulate the paths of all the images in the MNIST directory into their proper groups
    let paths = fs::read_dir(&TRAIN_DIR).unwrap();
    // let group_names = vec!["zero","one","two","three","four","five","six","seven","eight","nine"];
    let group_names = vec!["zero","one","two","three"];
    let mut groups: Vec<Vec<String>> = Vec::new();
    for group in &group_names {
        groups.push(vec![]);
    }
    for path in paths {
        let p = path.unwrap().path().to_str().unwrap().to_string();
        for i in 0..group_names.len() {
            if p.contains(group_names[i]) {
                groups[i].push(p.clone());
            }
        }
    }

    for group in &groups {
        if group.len() >= TRAIN_SET_SIZE {
            for i in 0..TRAIN_SET_SIZE {
                println!("{}",group[i]);
                let subpaths = group[i].split("/").collect::<Vec<&str>>();
                fs::copy(&group[i], CONSOLIDATED_TRAIN_DIR.clone() + "/" + subpaths.last().unwrap() ).unwrap();  // Copy foo.txt to bar.txt
            }
        }
    }

    // Let's accumulate the paths of all the images in the MNIST directory into their proper groups
    let paths = fs::read_dir(&TEST_DIR).unwrap();
    let group_names = vec!["one","two","three","four","five","six","seven","eight","nine"];
    let mut groups: Vec<Vec<String>> = Vec::new();
    for group in &group_names {
        groups.push(vec![]);
    }
    for path in paths {
        let p = path.unwrap().path().to_str().unwrap().to_string();
        for i in 0..group_names.len() {
            if p.contains(group_names[i]) {
                groups[i].push(p.clone());
            }
        }
    }

    for group in &groups {
        if group.len() >= TEST_SET_SIZE {
            for i in 0..TEST_SET_SIZE {
                println!("{}",group[i]);
                let subpaths = group[i].split("/").collect::<Vec<&str>>();
                fs::copy(&group[i], CONSOLIDATED_TEST_DIR.clone() + "/" + subpaths.last().unwrap() ).unwrap();  // Copy foo.txt to bar.txt
            }
        }
    }



}
