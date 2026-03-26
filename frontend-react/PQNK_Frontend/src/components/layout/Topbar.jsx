import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

export default function Topbar() {
  return (
    <div className="h-16 bg-white border-b flex justify-between items-center px-8">
      <h2 className="text-xl font-semibold">Dashboard</h2>

      <DropdownMenu>
        <DropdownMenuTrigger>
          <Avatar>
            <AvatarFallback>MW</AvatarFallback>
          </Avatar>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Profile</DropdownMenuItem>
          <DropdownMenuItem>Logout</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}