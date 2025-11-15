# On reviewing PRs
Thank you for your willingness to review PRs for Python 200! There is no single right way to review a PR: this is one set of guidelines to help you get started.

When you review a lesson (which will be a markdown file), we recommend you start by just reading as if you were a CTD student. You do not need deep subject-matter expertise, and you do not need to catch every formatting detail. Instead, in big picture terms, is the lesson clear, welcoming, and instructive? If so, let the person know! If not, start thinking about how we can improve the lesson.

For instance, does it slip into dense bullet lists and outline-style notes (this is often the default writing style of AI systems, and can be difficult for students to read and process). 

After the overall impression, we can start looking at some specifics. We like to include links to external resources in our lessons. Does the lesson includes links to videos or text resources for students, and do they seem accessible?

Also, images should be under 1 MB, in jpg or png format, and placed in the lesson's `resources/` folder. Any datasets or other supporting files should live in that same `resources` folder. We need to keep the repo small and tidy. :smile: 

Always assume good intent. Bear in mind someone poured time and heart into the work you are reviewing. Try to be positive. E.g., instead of saying "This is dry", you could say "Let's try to provide a motivating example here to show how this is important in industry." Better yet, providing such an example yourself in the review can be really helpful! 

A helpful review does not have to be *long*. Your first review can be big-picture, and focus on structural and design issues. As the PR gets closer to merge, you might zoom in on more details. 

Also, it is up to you whether to do an *official* review at GitHub by clicking Submit Review, or whether you would rather do it informally in slack. 

## Rendering the markdown at GitHub
GitHub doesn’t render `.md` files in the PR view--you’ll just see raw `# markdown` in the doc. 

As a way around this, to fully render the md:
1. Go to the `Files changed` tab  
2. Find the `.md` file you want to render (e.g., `04_llm_lesson.md`)  
3. Click the `...` in the top-right, and then select `View file` 

Now you will see the fully rendered markdown (this is how it will appear in the repo if it is merged). This is a great chance to spot broken images or formatting mishaps, which could be useful to mention in the review.

## First-time reviewer checklist
- [ ] Read through the lesson as if you were a beginner: is it clear and friendly, and resist the lure of bullet-point lists?  
- [ ] Does it include links to helpful external resources, including video and text?   
- [ ] Confirm all images are under 1 MB, in jpg or png format, and placed in `resources/`.    
- [ ] Confirm any datasets or extra files also live in the lesson's `resources/` folder.    
- [ ] Make sure images render correctly when viewing the file with "View file" in the PR (generally confirm that things are formatted correctly).    
- [ ] Leave review comments that are specific, encouraging, and constructive.   

Good luck -- even a short, thoughtful review makes a huge difference for a PR.  Thanks again for your willingness to help!