# AccessAble - Inclusive Professional Network

Welcome to AccessAble, a professional networking platform designed with accessibility at its core. Our goal is to create an inclusive space where everyone, regardless of their abilities, can connect, share, and grow their careers.

## Features

- **Accessible User Interface**: Designed with WCAG guidelines in mind, offering high contrast, large text options, and screen reader compatibility.
- **User Authentication**: Secure sign-up and login flows.
- **Personalized Profiles**: Create detailed profiles showcasing skills, experience, and accessibility needs.
- **Dynamic Feed**: View and interact with posts from your network.
- **Job Search**: Discover job opportunities with accessibility filters.
- **Messaging System**: Connect with other professionals.

## Getting Started

To run this project locally, follow these steps:

### 1. Clone the repository

\`\`\`bash
git clone [YOUR_REPOSITORY_URL]
cd pwd-linkedin
\`\`\`

### 2. Install dependencies

\`\`\`bash
npm install
# or
yarn install
\`\`\`

### 3. Run the development server

\`\`\`bash
npm run dev
# or
yarn dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

## Project Structure

- `app/`: Contains Next.js App Router pages and layouts.
- `components/`: Reusable React components, including UI components from `shadcn/ui`.
- `public/`: Static assets like images.
- `styles/`: Global CSS styles.
- `lib/`: Utility functions.
- `hooks/`: Custom React hooks.

## Accessibility Features

AccessAble is built with a strong focus on accessibility. Key features include:

- **High Contrast Mode**: Toggle for enhanced readability.
- **Large Text Mode**: Adjust font sizes for better visibility.
- **Audio Assistance Toggle**: Option to enable/disable audio cues.
- **Semantic HTML**: Proper use of HTML5 elements for better screen reader navigation.
- **ARIA Attributes**: Applied where necessary to improve accessibility for dynamic content.
- **Keyboard Navigation**: All interactive elements are navigable via keyboard.

## Contributing

We welcome contributions to make AccessAble even better! Please feel free to open issues or submit pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).
\`\`\`
