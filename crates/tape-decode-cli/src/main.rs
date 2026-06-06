mod cli;
mod decode;
mod fields_match;
mod flac;
mod metadata;
mod os;
mod profiles;
mod reader;
mod writer;

fn main() {
    if let Err(error) = cli::run_cli() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
