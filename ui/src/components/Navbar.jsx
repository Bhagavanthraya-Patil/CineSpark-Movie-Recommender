import { Link } from 'react-router-dom'
import { Search, User } from 'lucide-react'
import { useState } from 'react'

function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)

  window.onscroll = () => {
    setIsScrolled(window.scrollY > 0)
  }

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        isScrolled ? 'bg-netflix-black' : 'bg-gradient-to-b from-black/80 to-transparent'
      }`}
    >
      <div className="flex items-center justify-between px-8 py-4">
        <div className="flex items-center space-x-8">
          <Link to="/" className="text-2xl font-bold text-netflix-red">
            RagFlix
          </Link>
          <div className="hidden md:flex space-x-6">
            <Link to="/" className="hover:text-netflix-red transition-colors">
              Home
            </Link>
            <Link to="/search" className="hover:text-netflix-red transition-colors">
              Search
            </Link>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <Link to="/search">
            <Search className="w-6 h-6 hover:text-netflix-red transition-colors cursor-pointer" />
          </Link>
          <Link to="/profile">
            <User className="w-6 h-6 hover:text-netflix-red transition-colors cursor-pointer" />
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default Navbar

